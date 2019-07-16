const fs = require('fs');
const path = require('path');
const util = require('util');
const esprima = require('esprima');

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

async function main() {
    let list = (await find('./data')).filter(x => x.endsWith('.js'));
    list = list.slice(0, 1)
    await Promise.all(list.map(async file => {
        const content = await readFile(file, 'utf-8');
        console.log(content);
    }));
}

main()