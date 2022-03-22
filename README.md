# measfunc
<p align="center">
  <img src="https://img.shields.io/static/v1?style=for-the-badge&label=code-status&message=Caution!&color=red"/>
  <img src="https://img.shields.io/static/v1?style=for-the-badge&label=initial-commit&message=RasmusBC59&color=inactive"/>
    <img src="https://img.shields.io/static/v1?style=for-the-badge&label=maintainer&message=QDev&color=inactive"/>
</p>

# Description

MeasFunc is a repository for functions and classes used for doing measurements at QDev.
You are alowed to put (almost) everything in this repo. If a spcific function/class/projekt inside this repo matures we will migrate it to a seperatte repository.


# Installation
On a good day, installation is as easy as
```
$ git clone https://github.com/qdev-dk/MeasFunc.git
$ cd MeasFunc
$ pip install -e .
```

# Usage
If you have installed MeasFunc, and saved a file (ex. myfile.py) in the directory MeasFunc\measfunc\ containing a function (ex. myfunc) you can esaly import you function:

```python
from measfunc.myfile import myfunc
```

## Running the tests

If you have gotten 'measfunc' from source, you may run the tests locally.

Install `measfunc` along with its test dependencies into your virtual environment by executing the following in the root folder

```bash
$ pip install .
$ pip install -r test_requirements.txt
```

Then run `pytest` in the `tests` folder.

## Building the documentation

If you have gotten `measfunc` from source, you may build the docs locally.

Install `measfunc` along with its documentation dependencies into your virtual environment by executing the following in the root folder

```bash
$ pip install .
$ pip install -r docs_requirements.txt
```

You also need to install `pandoc`. If you are using `conda`, that can be achieved by

```bash
$ conda install pandoc
```
else, see [here](https://pandoc.org/installing.html) for pandoc's installation instructions.

Then run `make html` in the `docs` folder. The next time you build the documentation, remember to run `make clean` before you run `make html`.
