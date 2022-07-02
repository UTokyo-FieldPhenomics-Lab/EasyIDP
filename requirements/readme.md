# pip requirements files

## Index

- [default.txt](default.txt)
  Default requirements
- [docs.txt](docs.txt)
  Documentation requirements
- [test.txt](test.txt)
  Requirements for running test suite
- [build.txt](build.txt)
  Requirements for building from the source repository

## Examples

### Installing requirements

```bash
$ pip install -U -r requirements/default.txt
```

### Running the tests

```bash
$ pip install -U -r requirements/default.txt
$ pip install -U -r requirements/test.txt
```

> modified from https://github.com/scikit-image/scikit-image/blob/main/requirements/README.md