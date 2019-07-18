const fs = require('fs');
const path = require('path');
const util = require('util');
const esprima = require('esprima');
const walk = require( 'esprima-walk');
const createCsvWriter = require('csv-writer').createObjectCsvWriter;  

const readdir = util.promisify(fs.readdir);
const stat = util.promisify(fs.stat);
const readFile = util.promisify(fs.readFile);

async function find(dir) {
    return (await Promise.all(
        (await readdir(dir))
            .map(x => path.join(dir, x))
            .map(async x => (await stat(x)).isDirectory() ? await find(x) : x)
    )).reduce((x, y) => x.concat(y), []);
}

async function main(dir, out) {
    let list = (await find(dir)).filter(x => x.endsWith('.js'));
    const csvWriter = createCsvWriter({  
        path: out,
        header: [
            {id: 'name', title: 'name'},
            {id: 'api', title: 'api'},
            {id: 'token', title: 'token'},
            {id: 'desc', title: 'desc'},
        ]
    });
    for (let file of list) {
        const content = await readFile(file, 'utf-8');
        console.log('Analyzing ' + file);
        let ast;
        try {
            ast = esprima.parseModule(content, {loc:true, comment: true});
        } catch (err) {
            console.log('Error ' + file);
            continue;
        }
        const records = [];
        walk(ast, node => {
            if (node === null)
                return;
            let name = "", api = [], token = new Set(), desc = "";
            let body;
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
            })
            for (let comment of ast.comments) {
                if (comment.loc.end.line + 1 === node.loc.start.line) {
                    desc = comment.value;
                    break;
                }
            }
            records.push({
                name,
                api: api.join('|'),
                token: Array.from(token).join('|'),
                desc
            });
        });
        await csvWriter.writeRecords(records);
        console.log('Finished ' + file);
    }
}

main('/home/public/js_github_2', 'js2.csv');
// main('./data', 'js.csv');