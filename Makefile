js-origin-part1.csv: js-extract.js
	node $< /home/public/js_github $@

js-origin-part2.csv: js-extract.js
	node $< /home/public/js_github_2 $@

py-origin.csv: py-extract.py
	python3 $< /home/public/py_github $@

.PHONY: clean

clean:
	rm -f *.csv


