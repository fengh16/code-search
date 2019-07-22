const fs = require('fs');
const path = require('path');
const util = require('util');
const esprima = require('esprima');
const walk = require('esprima-walk');
const createCsvWriter = require('csv-writer').createObjectCsvWriter;
const process = require('process');

const readdir = util.promisify(fs.readdir);
const stat = util.promisify(fs.stat);
const readFile = util.promisify(fs.readFile);

let config_js = {
    "stop_words": ["a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are", "aren", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "couldn", "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn", "doesn't", "doing", "don", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't", "have", "haven", "haven't", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "ll", "m", "ma", "me", "mightn", "mightn't", "more", "most", "mustn", "mustn't", "my", "myself", "needn", "needn't", "no", "nor", "not", "now", "o", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan", "shan't", "she", "she's", "should", "should've", "shouldn", "shouldn't", "so", "some", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "won", "won't", "wouldn", "wouldn't", "y", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "could", "he'd", "he'll", "he's", "here's", "how's", "i'd", "i'll", "i'm", "i've", "let's", "ought", "she'd", "she'll", "that's", "there's", "they'd", "they'll", "they're", "they've", "we'd", "we'll", "we're", "we've", "what's", "when's", "where's", "who's", "why's", "would",
        "\r", "\n", ",", ".", "?", ";", ":", "'", "\"", "[", "]", "{", "}", "=", "+", "-", "*", "/", "\\", "&", "@", "$", "`", "!", "#", "%", "^", "(", ")", "|"
    ],
    "discard_remain": [
        "^(//)?(\\s)*todo",
        "(//)?(\\s)*['\"]use(\\s)+strict['\"]",
        "(//)?(\\s)*eslint-disable",
        "^(//)?(\\s)*export",
        "(//)?(\\s)*var ",
        "(.*)this code is generated(.*)"
    ],
    "discard_line": [
        "^(//)?(\\s)*@"
    ]
}

async function find(dir) {
    return (await Promise.all(
        (await readdir(dir))
            .map(x => path.join(dir, x))
            .map(async x => (await stat(x)).isDirectory() ? await find(x) : ((await stat(x)).isFile() ? x : []))
    )).reduce((x, y) => x.concat(y), []);
}

function checkDiscardRemain(c) {
    for (var temp of config_js['discard_remain']) {
        if (c.toLowerCase().search(temp) >= 0) {
            return true
        }
    }
    return false
}

function checkDiscardLine(c) {
    if (!isASCII(c)) {
        return true
    }
    for (var temp of config_js['discard_line']) {
        if (c.toLowerCase().search(temp) >= 0) {
            return true
        }
    }
    return false
}

function dealWithComment(beforeComment, afterComment) {
    let beforeComments = beforeComment.split('\n'), afterComments = afterComment.split('\n')
    let ans = ''
    for (var c of beforeComments) {
        if (checkDiscardRemain(c)) {
            break
        }
        if (checkDiscardLine(c.trim())) {
            continue
        }
        ans += c.trim() +'\n'
    }
    for (var c of afterComments) {
        if (checkDiscardRemain(c)) {
            break
        }
        if (checkDiscardLine(c.trim())) {
            continue
        }
        ans += c.trim() +'\n'
    }
    return ans.trim()
}

function isASCII(str) {
    return /^[\x00-\x7F]*$/.test(str);
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
            let lastCommentLine = -2
            let commentsOK = []
            let commentsStartLine = -2
            let beforeComment = ''
            let afterComment = ''
            for (let comment of ast.comments) {
                if (lastCommentLine + 1 === comment.loc.end.line) {
                    // 连续的
                    commentsOK.push(comment.value);
                    lastCommentLine = comment.loc.end.line
                }
                else if (commentsStartLine === node.loc.start.line + 1) {
                    // 不连续，函数声明这一行之后，说明已经找完了
                    beforeComment = commentsOK.join('\n');
                    break;
                }
                else {
                    // 不连续，新的一行
                    commentsStartLine = comment.loc.start.line
                    commentsOK = [comment.value]
                    lastCommentLine = comment.loc.end.line
                }
                if (comment.loc.end.line + 1 === node.loc.start.line) {
                    afterComment = commentsOK.join('\n');
                    commentsOK = []
                    lastCommentLine = -2
                }
            }
            desc = dealWithComment(beforeComment, afterComment)
            records.push({
                file,
                start: node.loc.start.line,
                end: node.loc.end.line,
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

if (process.argv.length !== 4)
    console.log('node js-extract.js <input_path> <output_file>')
else
    main(process.argv[2], process.argv[3])