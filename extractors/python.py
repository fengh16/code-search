#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ast
import glob
import csv
from os import path
import re
import argparse
from tqdm import tqdm
from collections import deque
import json

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
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input path that contains raw code')
    parser.add_argument('output', help='output csv that extracted data is written to')
    args = parser.parse_args()
    files = glob.glob(path.join(args.input, '**/*.py'), recursive=True)
    with open(args.output, "w") as out:
        f = csv.writer(out)
        f.writerow(['file', 'start', 'end', 'name', 'api', 'token', 'desc'])
        for file in tqdm(files):
            if not path.isfile(file):
                continue
            try:
                feature = extract(file)
            except:
                continue
            for item in feature:
                name, start, end, api, token, desc = item
                f.writerow((file, start, end, name,
                            '|'.join(api), '|'.join(token),
                            json.dumps([desc] if desc else [])))
