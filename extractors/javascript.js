#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const util = require('util');
const esprima = require('esprima');
const walk = require('esprima-walk');
const createCsvWriter = require('csv-writer').createObjectCsvWriter;
const yargs = require('yargs');
const ProgressBar = require('progress');

const readdir = util.promisify(fs.readdir);
const stat = util.promisify(fs.stat);
const readFile = util.promisify(fs.readFile);

async function find(dir) {
    return (await Promise.all(
        (await readdir(dir))
            .map(x => path.join(dir, x))
            .map(async x => {
                const type = await stat(x);
                if (type.isDirectory())
                    return await find(x);
                else if (type.isFile())
                    return x;
                else
                    return [];
            })
    )).reduce((x, y) => x.concat(y), []);
}

function getDocstring(comment) {
    comment_lines = comment.split('\n')
    let result = []
    for (let comment_line of comment_lines) {
        let trimed = comment_line.trim()
        if (trimed.indexOf('*') == 0) {
            line_data = trimed.slice(1).trim()
            if (line_data.indexOf('@') == 0) {
                console.log(result)
                return result.join('\n')
            }
            if (line_data.length > 0) {
                result.push(line_data)
            }
        }
        else {
            return ''
        }
    }
    return ''
}

async function main(dir, out) {
    let list = (await find(dir)).filter(x => x.endsWith('.js'));
    const csvWriter = createCsvWriter({  
        path: out,
        header: [
            {id: 'file', title: 'file'},
            {id: 'start', title: 'start'},
            {id: 'end', title: 'end'},
            {id: 'name', title: 'name'},
            {id: 'api', title: 'api'},
            {id: 'token', title: 'token'},
            {id: 'desc', title: 'desc'},
            {id: 'imported', title: 'imported'}
        ]
    });
    const bar = new ProgressBar(':bar :percent :current/:total [:elapsed<:eta, :rate it/s]', {total: list.length});
    for (let file of list) {
        const content = await readFile(file, 'utf-8');
        let ast;
        const imported_files = [];
        try {
            ast = esprima.parseModule(content, {loc:true, comment: true});
        } catch (err) {
            continue;
        }
        const comments = [];
        for (let comment of ast.comments) {
            if (comment.type === 'Block') {
                let docstring = getDocstring(comment.value)
                if (docstring) {
                    comments.push({
                        type: comment.type,
                        value: docstring,
                        start: comment.loc.start.line,
                        end: comment.loc.end.line
                    });
                }
            }
        }
        const records = [];
        walk(ast, node => {
            if (node === null)
                return;
            let name = "", api = [], token = new Set(), desc = "";
            let body;
            if (node.type === 'CallExpression') {
                if (node.callee.type === 'Identifier' && node.callee.name === 'require' && node.arguments.length === 1) {
                    if (node.arguments[0].value.indexOf('/') < 0) {
                        imported_files.push(node.arguments[0].value);
                    }
                }
            }
            else if (node.type == 'ImportDeclaration') {
                if (node.source.value.indexOf('/') < 0) {
                imported_files.push(node.source.value);
                node.specifiers.forEach(data => {if (data.type !== 'ImportNamespaceSpecifier') {
                    if (data.imported) {
                        imported_files.push(data.imported.name)
                    } else {
                        imported_files.push(data.local.name)}
                    }
                })
            }
        }
        switch (node.type) {
                case 'FunctionExpression':
                case 'ArrowFunctionExpression':
                case 'FunctionDeclaration':
                    if (node.id !== null)
                        name = node.id.name;
                    body = node.body;
                    break;
                case 'MethodDefinition':
                    if (node.key !== null && node.key.type === 'Identifier')
                        name = node.key.name;
                    body = node.value;
                default:
                    return;
            }
            walk(body, node => {
                if (node === null)
                    return;
                switch (node.type) {
                    case 'Identifier':
                        token.add(node.name);
                        break;
                    case 'CallExpression':
                        let tree = node.callee, callee = [];
                        while (tree.type === 'MemberExpression') {
                            if (tree.property.type === 'Identifier') {
                                callee.unshift(tree.property.name);
                                tree = tree.object;
                            } else
                                break;
                        }
                        if (tree.type === 'Identifier')
                            callee.unshift(tree.name);
                        api.push(callee.join('.'));
                    default:
                        break;
                }
            });
            let beforeComment, afterComment;
            for (let comment of comments) {
                if (comment.end + 1 === node.loc.start.line)
                    beforeComment = comment.value;
                else if (node.loc.start.line + 1 === comment.start)
                    afterComment = comment.value;
            }
            records.push({
                file,
                start: node.loc.start.line,
                end: node.loc.end.line,
                name,
                api: api.join('|'),
                token: Array.from(token).join('|'),
                desc: JSON.stringify([beforeComment, afterComment].filter(Boolean)),
                imported: imported_files
            });
        });
    records.forEach(record => record.imported = record.imported.join('|'))
        await csvWriter.writeRecords(records);
        bar.tick();
    }
}

const argv = yargs
    .usage('$0 <input_path> <output_file>')
    .help()
    .argv;
if (argv._.length === 2)
    main(argv._[0], argv._[1])
        .catch(e => console.error(e));
else
    yargs.showHelp();
