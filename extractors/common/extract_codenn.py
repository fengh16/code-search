import ast
import re
from collections import deque

class CallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.names = deque()

    def visit_Name(self, node):
        self.names.appendleft(node.id)

    def visit_Attribute(self, node):
        try:
            self.names.appendleft(node.attr)
            self.names.appendleft(node.value.id)
        except AttributeError:
            self.generic_visit(node)

def extract(content):
    module = ast.parse(content)
    lines = content.split('\n')
    result = []
    package_names = []
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef):
            desc = ast.get_docstring(node)
            end = node.body[0].lineno - 1
            if desc is not None:
                end += 1
            prefix_groups = re.match(r'^(\s+)\w', lines[end])
            while end + 1 < len(lines) and prefix_groups is None:
                end += 1
                prefix_groups = re.match(r'^(\s+)\w', lines[end])
            if end + 1 == len(lines):
                end = node.body[0].lineno - 1
            else:
                prefix = prefix_groups.group(1)
                end += 1
                while end < len(lines) and (not lines[end] or
                        lines[end].isspace() or lines[end].startswith(prefix)):
                    end += 1
                end -= 1
                while end >= 0 and (not lines[end] or lines[end].isspace()):
                    end -= 1
            name = node.name
            api = []
            token = set()
            for subnode in ast.walk(node):
                if isinstance(subnode, ast.Name):
                    token.add(subnode.id)
                elif isinstance(subnode, ast.Attribute):
                    token.add(subnode.attr)
                elif isinstance(subnode, ast.Call):
                    visitor = CallVisitor()
                    visitor.visit(subnode.func)
                    api.append('.'.join(visitor.names))
            result.append([name, node.lineno, end + 1, api, token, desc])
        if isinstance(node, ast.Import):
            for name in node.names:
                package_names.append(name.name)
    result = [tuple(r + [package_names]) for r in result]
    return result
