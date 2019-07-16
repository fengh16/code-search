import ast
import glob
import csv
from os import path
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
        module = ast.parse(f.read(), filename=file)
    result = []
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef):
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
            result.append((name, api, token, desc))
    return result

files = glob.glob('/home/public/py_github/**/*.py', recursive=True)

with open("result.csv", "w") as out:
    f = csv.writer(out)
    f.writerow(['file', 'name', 'api', 'token', 'desc'])
    for file in files:
        if not path.isfile(file):
            continue
        print('Processing ' + file)
        try:
            feature = extract(file)
            for item in feature:
                name, api, token, desc = item
                f.writerow((file, name, '|'.join(api), '|'.join(token), desc))
        except:
            pass
