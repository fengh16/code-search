py-origin.csv: py-extract.py
	python3 $< /home/public/py_github $@

js-origin-part1.csv: js-extract.js
	node $< /home/public/js_github $@

js-origin-part2.csv: js-extract.js
	node $< /home/public/js_github_2 $@

js-origin.csv: js-origin-part1.csv js-origin-part2.csv
	cat js-origin-part1.csv > $@
	cat js-origin-part2.csv | tail -n +2 >> $@

py-cleaned.csv: py-clean.py py-origin.csv
	python3 $< py-origin.csv $@

js-cleaned.csv: js-clean.py js-origin.csv
	python3 $< js-origin.csv $@

.PHONY: clean

clean:
	rm -f *.csv


