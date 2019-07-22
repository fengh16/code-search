import ast
import glob
import csv
from os import path
import re
import sys
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

def extract(file):
    with open(file) as f:
        content = f.read()
        module = ast.parse(content, filename=file)
        lines = content.split('\n')
    result = []
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef):
            end = node.body[0].lineno - 1
            prefix = re.match('^[ \t]+', lines[end]).group(0)
            end += 1
            while end < len(lines) and (not lines[end] or lines[end].startswith(prefix)):
                end += 1
            end -= 1
            while end >= 0 and (not lines[end] or lines[end].isspace()):
                end -= 1
            name = node.name
            api = []
            token = set()
            desc = ast.get_docstring(node)
            for subnode in ast.walk(node):
                if isinstance(subnode, ast.Name):
                    token.add(subnode.id)
                elif isinstance(subnode, ast.Attribute):
                    token.add(subnode.attr)
                elif isinstance(subnode, ast.Call):
                    visitor = CallVisitor()
                    visitor.visit(subnode.func)
                    api.append('.'.join(visitor.names))
            result.append((name, node.lineno, end + 1, api, token, desc))
    return result


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("py-extract.py <input_path> <output_file>")
        sys.exit()
    files = glob.glob(path.join(sys.argv[1], '**/*.py'), recursive=True)
    with open(sys.argv[2], "w") as out:
        f = csv.writer(out)
        f.writerow(['file', 'start', 'end', 'name', 'api', 'token', 'desc'])
        for file in files:
            if not path.isfile(file):
                continue
            print('Analyzing ' + file)
            try:
                feature = extract(file)
            except:
                print('Error ' + file)
                continue
            for item in feature:
                name, start, end, api, token, desc = item
                f.writerow((file, start, end, name,
                            '|'.join(api), '|'.join(token), desc))
            print('Finished ' + file)
