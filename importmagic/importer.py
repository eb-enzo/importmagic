"""Imports new symbols."""
import itertools
import tokenize
from collections import defaultdict

from importmagic.six import StringIO

LINE_FEED = '\n'
IMPORT_TYPE_FROM = 'from'
IMPORT_TYPE_IMPORT = 'import'
IMPORT_TYPES = frozenset([IMPORT_TYPE_FROM, IMPORT_TYPE_IMPORT])


class Iterator(object):
    def __init__(self, tokens, start=None, end=None):
        self._tokens = tokens
        self._cursor = start or 0
        self._end = end or len(self._tokens)

    def rewind(self):
        self._cursor -= 1

    def next(self):
        if not self:
            return None, None
        token = self._tokens[self._cursor]
        index = self._cursor
        self._cursor += 1
        return index, token

    def peek(self):
        return self._tokens[self._cursor] if self else None

    def until(self, type):
        tokens = []
        while self:
            index, token = self.next()
            tokens.append((index, token))
            if type == token[0]:
                break
        return tokens

    def __nonzero__(self):
        return self._cursor < self._end
    __bool__ = __nonzero__


class Import(object):
    def __init__(self, location, name, alias, import_type):
        self.location = location
        self.name = name
        self.alias = alias
        self.import_type = import_type

    def __repr__(self):
        return 'Import(location=%r, name=%r, alias=%r)' % \
            (self.location, self.name, self.alias)

    def __hash__(self):
        return hash((self.location, self.name, self.alias))

    def __eq__(self, other):
        return self.location == other.location and self.name == other.name and self.alias == other.alias

    def __ne__(self, other):
        return not (self == other)

    def __lt__(self, other):
        return self.location < other.location \
            or self.name < other.name \
            or (self.alias is not None and other.alias is not None and self.alias < other.alias)


class ImportSubBlock(object):

    def __init__(self):
        self.imports_dict = defaultdict(list)

    def add_imports(self, module, import_to_add):
        self.imports_dict[module].append(import_to_add)

    def serialize(self):
        out = StringIO()
        modules = sorted(self.imports_dict.keys())
        for module in modules:
            imports = sorted(self.imports_dict[module])
            if len(imports) == 1 and imports[0].import_type == IMPORT_TYPE_IMPORT:
                imp = imports[0]
                line = 'import {module}{alias}'.format(
                    module=module,
                    alias=' as {alias}'.format(alias=imp.alias) if imp.alias else ''
                )
                clauses = ['']
            else:
                line = 'from {module} import '.format(module=module)
                clauses = ['{name}{alias}'.format(
                    name=i.name,
                    alias=' as {alias}'.format(alias=i.alias) if i.alias else ''
                ) for i in imports]

            if len(clauses) == 1:
                line = line + clauses[0] + LINE_FEED
            else:
                clauses_lines = (',' + LINE_FEED).join([
                    ' '*4 + clause for clause in clauses
                ])
                line += '(\n{},\n)\n'.format(clauses_lines)

            if line.strip():
                out.write(line)
        return out.getvalue()

    def is_empty(self):
        return not self.imports_dict


class ImportBlock(object):
    def serialize(self):
        out = StringIO()
        out.write(
            '\n'.join(
                sub_block.serialize() for sub_block in self.sub_blocks if not sub_block.is_empty()
            )
        )
        return out.getvalue()


class HeaderImportBlock(ImportBlock):

    def __init__(self):
        self.libs = ImportSubBlock()
        self.django = ImportSubBlock()

    def add_import(self, main_module, import_to_add):
        if main_module == 'django':
            self.django.add_imports(main_module, import_to_add)
        else:
            self.libs.add_imports(main_module, import_to_add)

    @property
    def sub_blocks(self):
        return [self.libs, self.django]


class BodyImportBlock(ImportBlock):
    associated_libs = [
        'soa',
        'service',
        'eb',
        'common',
    ]

    def __init__(self):
        self.sub_blocks_map = defaultdict(ImportSubBlock)

    def add_import(self, main_module, import_to_add):
        self.sub_blocks_map[main_module.split('.')[0]].add_imports(main_module, import_to_add)

    @property
    def sub_blocks(self):
        keys = self.sub_blocks_map.keys()
        keys.sort()
        return [self.sub_blocks_map[key] for key in keys]


class FooterImportBlock(ImportBlock):

    def __init__(self):
        self.footer_block = ImportSubBlock()

    def add_import(self, main_module, import_to_add):
        self.footer_block.add_imports(main_module, import_to_add)

    @property
    def sub_blocks(self):
        return [self.footer_block]


def block_for(module):
    if module.startswith('.'):
        return '_footer'
    elif any(module.startswith(eblib) for eblib in BodyImportBlock.associated_libs):
        return '_body'
    else:
        return '_header'


# See SymbolIndex.LOCATIONS for details.
LOCATION_ORDER = 'FS3L'


class Imports(object):

    _style = {
        'multiline': 'parentheses',
        'max_columns': 120,
    }

    def __init__(self, index, source):
        self._header = HeaderImportBlock()
        self._body = BodyImportBlock()
        self._footer = FooterImportBlock()
        self._imports = set()
        self._imports_from = defaultdict(set)
        self._imports_begin = self._imports_end = None
        self._source = source
        self._index = index
        self._parse(source)

    @classmethod
    def set_style(cls, **kwargs):
        cls._style.update(kwargs)

    @property
    def all_imports(self):
        return itertools.chain(self._header, self._body, self._footer)

    def add_import(self, name, alias=None):
        location = LOCATION_ORDER.index(self._index.location_for(name))
        location_new = block_for(name)

        getattr(self, location_new).add_import(name, Import(location, name, alias, IMPORT_TYPE_IMPORT))

    def add_import_from(self, module, name, alias=None):
        location = LOCATION_ORDER.index(self._index.location_for(module))
        location_new = block_for(module)
        getattr(self, location_new).add_import(module, Import(location, name, alias, IMPORT_TYPE_FROM))

    def remove(self, references):
        for imp in list(self._imports):
            if imp.name in references:
                self._imports.remove(imp)
        for name, imports in self._imports_from.items():
            for imp in list(imports):
                if imp.name in references:
                    imports.remove(imp)

    def get_update(self):
        out = StringIO()
        out.write(
            '\n'.join(
                block.serialize() for block in (self._header, self._body, self._footer,)
            )
        )
        out.write('\n')
        text = out.getvalue()
        start = self._tokens[self._imports_begin][2][0] - 1
        end = self._tokens[min(len(self._tokens) - 1, self._imports_end)][2][0] - 1

        return start, end, text

    def update_source(self):
        start, end, text = self.get_update()
        lines = self._source.splitlines()
        lines[start:end] = text.splitlines()
        return '\n'.join(lines) + '\n'

    def _parse(self, source):
        reader = StringIO(source)
        # parse until EOF or TokenError (allows incomplete modules)
        tokens = []
        try:
            tokens.extend(tokenize.generate_tokens(reader.readline))
        except tokenize.TokenError:
            # TokenError happens always at EOF, for unclosed strings or brackets.
            # We don't care about that here, since we still can recover the whole
            # source code.
            pass
        self._tokens = tokens
        it = Iterator(self._tokens)
        self._imports_begin, self._imports_end = self._find_import_range(it)
        it = Iterator(self._tokens, start=self._imports_begin, end=self._imports_end)
        self._parse_imports(it)

    def _find_import_range(self, it):
        ranges = self._find_import_ranges(it)
        start, end = ranges[0][1:]
        return start, end

    def _find_import_ranges(self, it):
        ranges = []
        indentation = 0
        explicit = False
        size = 0
        start = None
        potential_end_index = -1

        while it:
            index, token = it.next()

            if token[0] == tokenize.INDENT:
                indentation += 1
                continue
            elif token[0] == tokenize.DEDENT:
                indentation += 1
                continue

            if indentation:
                continue

            # Explicitly tell importmagic to manage the following block of imports
            if token[1] == '# importmagic: manage':
                ranges = []
                start = index + 2  # Start managing imports after directive comment + newline.
                explicit = True
                continue
            elif token[0] in (tokenize.STRING, tokenize.COMMENT):
                # If a non-import statement follows, stop the range *before*
                # this string or comment, in order to keep it out of the
                # updated import block.
                if potential_end_index == -1:
                    potential_end_index = index
                continue
            elif token[0] in (tokenize.NEWLINE, tokenize.NL):
                continue

            if not ranges:
                ranges.append((0, index, index))

            # Accumulate imports
            if token[1] in IMPORT_TYPES:
                potential_end_index = -1
                if start is None:
                    start = index
                size += 1
                while it:
                    token = it.peek()
                    if token[0] == tokenize.NEWLINE or token[1] == ';':
                        break
                    index, _ = it.next()

            # Terminate this import range
            elif start is not None and token[1].strip():
                if potential_end_index > -1:
                    index = potential_end_index
                    potential_end_index = -1
                ranges.append((size, start, index))

                start = None
                size = 0
                if explicit:
                    break

        if start is not None:
            ranges.append((size, start, index))
        ranges.sort(reverse=True)
        return ranges

    def _parse_imports(self, it):
        while it:
            index, token = it.next()

            if token[1] not in IMPORT_TYPES and token[1].strip():
                continue

            type = token[1]
            if type in IMPORT_TYPES:
                tokens = it.until(tokenize.NEWLINE)
                tokens = [t[1] for i, t in tokens
                          if t[0] == tokenize.NAME or t[1] in ',.']
                tokens.reverse()
                self._parse_import(type, tokens)

    def _parse_import(self, type, tokens):
        module = None
        if type == IMPORT_TYPE_FROM:
            module = ''
            while tokens and tokens[-1] != 'import':
                module += tokens.pop()
            assert tokens.pop() == 'import'

        while tokens:
            name = ''
            while True:
                name += tokens.pop()
                next = tokens.pop() if tokens else None
                if next == '.':
                    name += next
                else:
                    break

            alias = None
            if next == 'as':
                alias = tokens.pop()
                if alias == name:
                    alias = None
                next = tokens.pop() if tokens else None
            if next == ',':
                pass
            if type == IMPORT_TYPE_IMPORT:
                self.add_import(name, alias=alias)
            else:
                self.add_import_from(module, name, alias=alias)

    def __repr__(self):
        return 'Imports(imports=%r, imports_from=%r)' % (self._imports, self._imports_from)


def _process_imports(src, index, unresolved, unreferenced):
    imports = Imports(index, src)
    imports.remove(unreferenced)
    for symbol in unresolved:
        scores = index.symbol_scores(symbol)
        if not scores:
            continue
        _, module, variable = scores[0]
        # Direct module import: eg. os.path
        if variable is None:
            # sys.path              sys path          ->    import sys
            # os.path.basename      os.path basename  ->    import os.path
            imports.add_import(module)
        else:
            # basename              os.path basename   ->   from os.path import basename
            # path.basename         os.path basename   ->   from os import path
            imports.add_import_from(module, variable)
    return imports


def get_update(src, index, unresolved, unreferenced):
    imports = _process_imports(src, index, unresolved, unreferenced)
    return imports.get_update()


def update_imports(src, index, unresolved, unreferenced):
    imports = _process_imports(src, index, unresolved, unreferenced)
    return imports.update_source()
