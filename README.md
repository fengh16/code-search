# Code Search

## About

This project is intended to search code with natural language.

## Layout

- `data`: contains all the data
  - `rawcode`: contains folder that contains the source code
  - `extracted`: contains csvs for extracted dirty data
  - `cleaned`: contains csvs that is cleaned
  - `transformed`: contains folder whose inside files are suitable for training
- `extractors`: scripts used to extract data from raw code

## How to Run This Project

### 1 Prepare Data

Please put source code under `data/raw`. The root of the source code should be exactly one folder inside `data/raw`.

### 2 Extract all Comments/Docstrings and Code Snippet Features

Currently only Python and JavaScript are supported.

```bash
./extractor --extract {python,javascript} --dataset <dataset>
```

Replace `<dataset>` with the root directory name of you data.