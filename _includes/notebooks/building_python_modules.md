### Getting Started with Python Packaging

The first two weeks of the state-space project have been dedicated to introducing the Kalman filter - which was written in Cython with calls to the BLAS and LAPACK libraries linked to in Scipy - into the Statsmodels build process. A future post may describe why it was not just written in pure Python (in brief, it is because the Kalman filter is a recursive algorithm with a loop over the number of entries in a dataset, where each loop involves many matrix operations on relatively small matrices). For now, though, the source `kalman_filter.pyx` needs to be "Cythonized" into `kalman_filter.c` and then compiled into (e.g.) `kalman_filter.so`, either when the package is installed using pip, or from source (e.g. `python setup.py install`).

The first thing to figure out was the state of Python's packaging. I've had a vague sense of some of the various tools of Python packaging for a while (especially since it used to be recommended to specify `--distribute` which making a new [virtualenv](http://www.google.com)), but I built all my Cython packages either via a `python setup.py build_ext --inplace` (from the [Cython quickstart](http://docs.cython.org/src/quickstart/build.html)) or via [IPython magic](http://ipython.org/ipython-doc/2/config/extensions/cythonmagic.html).

The recommended `setup.py` file from Cython quickstart is:

```python
from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'Hello world app',
  ext_modules = cythonize("hello.pyx"),
)
```

as you can see, this uses the [`distutils`](https://docs.python.org/2/library/distutils.html) package. However, while `distutils` is part of base Python and is standard for packaging, it, from what I could tell, `distribute` was the up-and-coming way to proceed. Would that it were that simple; it turns out that Python packaging is not for the faint of heart. A wonderful [stackoverflow answer](http://stackoverflow.com/a/14753678/603962) describes the state-of-the-art (hopefully) as of October 2013. It comes to the conclusion that [`setuptools`](https://pythonhosted.org/setuptools/) is probably the way to go, unless you only need basic packaging, in which case you should use `distutils`.

### Setuptools

So it appeared that the way to go was to use `setuptools` (and more than personal preference, Statsmodels [uses `setuptools`](https://github.com/statsmodels/statsmodels/blob/master/setup.py#L30)). Unfortunately, I have always previously used the above snippet which is `distutils` based, and as it turns out, the magic that makes that bit of code possible *is not available in setuptools*. You can read [this mailing list conversation](https://mail.python.org/pipermail/distutils-sig/2007-September/008207.html) from September 2013 for a fascinating back-and-forth about what should be supported where, leading to the surprising conclusion that to make Setuptools automatically call Cython to build `*.pyx` files, one should *trick* it into believing there was a fake Pyrex installation.

This approach can be seen at the [repository](http://github.com/ChadFulton/pykalman_filter) for the existing Kalman filter code, or at https://github.com/njsmith/scikits-sparse (in both cases, look for the "fake_pyrex" directory in the project root).

It's often a good idea, though, to look at [NumPy](http://github.com/numpy/numpy) and [SciPy](https://github.com/scipy/scipy) for *how it should be done*, and it turns out that neither of them use a fake Pyrex directory, and neither do rely on `setuptools` (or `distutils`) to Cythonize the `*.pyx` files. Instead, they use a direct `subprocess` call to `cythonize` directly. Why do this, though?

### NumPy and SciPy

Although at first it seemed like an awfully Byzantine and arbitrary mish-mash of roll-your-owns, where no two parties do things the same way, it turns out that the NumPy / SciPy approach agrees, in spirit, with the latest `Cython` [documentation on compilation](http://docs.cython.org/src/reference/compilation.html). The idea is that `Cython` should not be a required dependency for installation, and thus the *already Cythonized* `*.c` files should be included in the distributed package. These will be cythonized during the `python setup.py sdist` process.

So the end result is that setuptools should not be required to cythonize `*.pyx` files, it only needs to compile and link `*.c` files (which it has no problem with - no fake pyrex directory needed). Then the question is, how to cythonize the files? It turns out that the common way, as mentioned above, is to use a subprocess call to the `cythonize` binary directly (see [Statsmodels](https://github.com/statsmodels/statsmodels/blob/master/setup.py#L86), [NumPy](https://github.com/numpy/numpy/blob/master/setup.py#L187), [SciPy](https://github.com/scipy/scipy/blob/master/setup.py#L158)).
