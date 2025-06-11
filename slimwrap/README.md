# slimwrap

#### Author: Curro Campuzano

## Motivation

SLiMGui is a fantastic tool for developing and debugging SLiM code. However, I find myself often in one of the following scenarios:

- Sharing a complete analysis workflow (including preparing some data, simulating data with SLiM, analyzing, and visualizing it). It is handy being able to share a self-contained Jupyter notebook with text, formulas, all the necessary code, and some plots.
- Running hundreds of replicates of a simulation that involves many steps that rely on multiple programs. Often, the necessary glue code to manage it is easier to develop outside SLiM (in a high-level language). However, this requires writing repetitive boilerplate code to pass information to and from SLiM.

This project relies on the fact that it is possible to “inject” and execute arbitrary code into SLiM execution via the `-d` flag. So far, this wrapper is only implemented in Python, although it could be extended to other languages.

## slimwrap.py

The wrapper is a single-file Python module (located at [python/slimwrap.py](python/slimwrap.py)) with minimal dependencies that you can copy-paste into your project. You can take a look at a more comprehensive description of the module in [python/example.ipynb](python/example.ipynb).

The library works by encoding data into a temporary JSON file and performing some additional parsing to load primitives (logical, floats, etc.) and 1D and 2D arrays. Additionally, it allows you to define SLiM models within a Jupyter notebook and share them (with some extra features like syntax highlighting via JavaScript). It is, however, intended to be a thin wrapper around the `subprocess` and `json` standard library.

You can also find an alternative approach to this problem in `sublaunching_tutorial` directory.

# Development

The test suite is located at [python/test_slimwrap.py](python/test_slimwrap.py). To run the tests, execute the following command:

```
python -m pytest python/test_slimwrap.py
```
