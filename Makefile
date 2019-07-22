all: data/py/vocab.desc.pkl data/js/vocab.desc.pkl

data/origin/py-origin.csv: py-extract.py
	python3 $< /home/public/py_github $@

data/origin/js-origin-part1.csv: js-extract.js
	node $< /home/public/js_github $@

data/origin/js-origin-part2.csv: js-extract.js
	node $< /home/public/js_github_2 $@

data/origin/js-origin.csv: data/origin/js-origin-part1.csv data/origin/js-origin-part2.csv
	cat js-origin-part1.csv > $@
	cat js-origin-part2.csv | tail -n +2 >> $@

data/origin/py-cleaned.csv: clean.py data/origin/py-origin.csv
	python3 $< data/origin/py-origin.csv $@

data/origin/js-cleaned.csv: clean.py data/origin/js-origin.csv
	python3 $< data/origin/js-origin.csv $@

data/py/vocab.desc.pkl: transform.py data/origin/py-cleaned.csv
	python3 $< data/origin/py-cleaned.csv data/py

data/js/vocab.desc.pkl: transform.py data/origin/js-cleaned.csv
	python3 $< data/origin/js-cleaned.csv data/js

.PHONY: clean all

clean:
	rm -rf data


