"""
Function extraction from C source files using tree-sitter.
"""

import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

try:
    import tree_sitter_c as tsc
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False


@dataclass
class CFunction:
    """Represents an extracted C function."""
    function_name: str
    function_code: str
    file_path: str
    start_line: int
    end_line: int

    def to_dict(self) -> dict:
        return asdict(self)


class CFunctionExtractor:
    """Extracts functions from C source files using tree-sitter."""

    def __init__(self):
        if TREE_SITTER_AVAILABLE:
            self.parser = Parser(Language(tsc.language()))
        else:
            self.parser = None

    def extract_from_file(self, file_path: str) -> List[CFunction]:
        """Extract all functions from a C file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        code = path.read_text(encoding='utf-8', errors='ignore')

        if self.parser:
            return self._extract_with_tree_sitter(code, file_path)
        else:
            return self._extract_with_regex(code, file_path)

    def _extract_with_tree_sitter(self, code: str, file_path: str) -> List[CFunction]:
        """Extract functions using tree-sitter AST parsing."""
        tree = self.parser.parse(bytes(code, 'utf-8'))
        functions = []

        def traverse(node):
            if node.type == 'function_definition':
                # Get function name from declarator
                func_name = self._get_function_name(node)
                if func_name:
                    start_line = node.start_point[0] + 1
                    end_line = node.end_point[0] + 1
                    func_code = code[node.start_byte:node.end_byte]

                    functions.append(CFunction(
                        function_name=func_name,
                        function_code=func_code,
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line
                    ))

            for child in node.children:
                traverse(child)

        traverse(tree.root_node)
        return functions

    def _get_function_name(self, node) -> Optional[str]:
        """Extract function name from a function_definition node."""
        for child in node.children:
            if child.type == 'function_declarator':
                for subchild in child.children:
                    if subchild.type == 'identifier':
                        return subchild.text.decode('utf-8')
            elif child.type == 'pointer_declarator':
                # Handle pointer return types like: int *func()
                return self._get_function_name_from_declarator(child)
        return None

    def _get_function_name_from_declarator(self, node) -> Optional[str]:
        """Recursively find function name in nested declarators."""
        for child in node.children:
            if child.type == 'function_declarator':
                for subchild in child.children:
                    if subchild.type == 'identifier':
                        return subchild.text.decode('utf-8')
            elif child.type in ('pointer_declarator', 'parenthesized_declarator'):
                result = self._get_function_name_from_declarator(child)
                if result:
                    return result
        return None

    def _extract_with_regex(self, code: str, file_path: str) -> List[CFunction]:
        """Fallback: Extract functions using regex patterns."""
        functions = []

        # Pattern to match C function definitions
        # Matches: return_type function_name(params) { ... }
        pattern = r'''
            (?:^|\n)                           # Start of line
            \s*                                # Optional whitespace
            (?:static\s+|inline\s+|extern\s+)* # Optional qualifiers
            (?:const\s+)?                      # Optional const
            [\w\s\*]+?                         # Return type (word chars, spaces, pointers)
            \s+                                # Whitespace
            (\w+)                              # Function name (captured)
            \s*\(                              # Opening parenthesis
            [^)]*                              # Parameters
            \)\s*                              # Closing parenthesis
            \{                                 # Opening brace
        '''

        lines = code.split('\n')

        for match in re.finditer(pattern, code, re.VERBOSE | re.MULTILINE):
            func_name = match.group(1)
            start_pos = match.start()

            # Find the matching closing brace
            brace_start = match.end() - 1
            end_pos = self._find_matching_brace(code, brace_start)

            if end_pos > brace_start:
                func_code = code[match.start():end_pos + 1].strip()
                start_line = code[:start_pos].count('\n') + 1
                end_line = code[:end_pos].count('\n') + 1

                functions.append(CFunction(
                    function_name=func_name,
                    function_code=func_code,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line
                ))

        return functions

    def _find_matching_brace(self, code: str, start: int) -> int:
        """Find the position of the matching closing brace."""
        depth = 0
        in_string = False
        in_char = False
        in_comment = False
        in_line_comment = False
        i = start

        while i < len(code):
            c = code[i]

            # Handle line comments
            if in_line_comment:
                if c == '\n':
                    in_line_comment = False
                i += 1
                continue

            # Handle block comments
            if in_comment:
                if c == '*' and i + 1 < len(code) and code[i + 1] == '/':
                    in_comment = False
                    i += 2
                    continue
                i += 1
                continue

            # Check for comment start
            if c == '/' and i + 1 < len(code):
                if code[i + 1] == '/':
                    in_line_comment = True
                    i += 2
                    continue
                elif code[i + 1] == '*':
                    in_comment = True
                    i += 2
                    continue

            # Handle strings
            if c == '"' and not in_char:
                if not in_string:
                    in_string = True
                elif i > 0 and code[i - 1] != '\\':
                    in_string = False
                i += 1
                continue

            # Handle char literals
            if c == "'" and not in_string:
                if not in_char:
                    in_char = True
                elif i > 0 and code[i - 1] != '\\':
                    in_char = False
                i += 1
                continue

            if in_string or in_char:
                i += 1
                continue

            # Count braces
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return i

            i += 1

        return start  # No matching brace found


def extract_control_flow(func_code: str) -> List[str]:
    """
    Extract control flow elements deterministically via regex.
    This is more reliable than using a classifier for these elements.
    """
    CONTROL_FLOW_KEYWORDS = ["if", "for", "while", "switch", "goto", "return"]
    found = []

    for kw in CONTROL_FLOW_KEYWORDS:
        # Match keyword as whole word (not part of identifier)
        if re.search(rf'\b{kw}\b', func_code):
            found.append(kw)

    return found


def extract_functions_from_directory(directory: str) -> List[CFunction]:
    """Extract all functions from all C files in a directory."""
    extractor = CFunctionExtractor()
    all_functions = []

    path = Path(directory)
    for c_file in path.rglob('*.c'):
        try:
            functions = extractor.extract_from_file(str(c_file))
            all_functions.extend(functions)
        except Exception as e:
            print(f"Warning: Failed to parse {c_file}: {e}")

    return all_functions


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python extract.py <c_file_or_directory>")
        sys.exit(1)

    path = Path(sys.argv[1])
    extractor = CFunctionExtractor()

    if path.is_file():
        functions = extractor.extract_from_file(str(path))
    else:
        functions = extract_functions_from_directory(str(path))

    print(f"Extracted {len(functions)} functions")
    for func in functions:
        print(f"  - {func.function_name} ({func.file_path}:{func.start_line}-{func.end_line})")
