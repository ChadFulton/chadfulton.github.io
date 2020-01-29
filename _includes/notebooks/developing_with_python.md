Python on Mac OS X has traditionally been a bit of a struggle, dealing with paths etc. to make sure the built-in Python (out of date) doesn't conflict with the Python I want to use (e.g. from Homebrew). Further complicating this, I had at various points installed Enthought EPD and Enthought Canopy. I'd like the state of Python on my computer to be less haphazard. To make that happen, I need to uninstall all Enthought products, and get everything cleaned out with respect to Homebrew, Xcode, and Virtualenv.

<!-- TEASER_END -->

### Cleaning / Updates

In the case of Homewbrew and Xcode, I just need to upgrade the packages and make sure everything is installed correctly. My Macbook Air has limited hard drive space, so I install the Xcode command line tools only. In the case of homebrew ```brew upgrade``` mostly worked, but something I'd done had left the libpng and freetype2 packages with problems:

    Error: You must `brew link libpng' before qt can be installed
    Warning: Could not link qt. Unlinking...
    Error: You must `brew link libpng' before pyqt can be installed
    Warning: Could not link pyqt. Unlinking...
    Error: You must `brew link libpng' before shiboken can be installed
    Warning: Could not link shiboken. Unlinking...
    Error: You must `brew link libpng' before pyside can be installed
    Warning: Could not link pyside. Unlinking...
    Error: You must `brew link freetype' before fontconfig can be installed
    Error: You must `brew link freetype' before pango can be installed

The problem was that the links already existed, but were owned by root, and even with the ```brew link --overwrite libpng``` syntax, it wouldn't work. The next guess is to use ```sudo```, but homebrew doesn't like using sudo. Ultimately, I used ```brew link --overwrite --dry-run libpng``` to identify the files that needed to be overwritten, ```chown```-ed them, performed the linking, and then ```chown```-ed them back to root. I doubt this was the best solution, but it worked.

To clean up, I removed all my old virtual environments and uninstalled homebrew's python, and I cleaned up the /usr/local/share/python folder, which had some out-of-date scripts in there. Then I reinstalled homebrew's python, and created a new virtual environment.

### ScipySuperpack

Aside from installing one of the pre-built python packages like [Anaconda][], [Enthought Canopy][], etc., one of the best ways to install the scientific computing libraries for Python (numpy, scipy, etc.) on Mac OS X is using Chris Fonnesbeck's excellent [ScipySuperpack][].

Fortunately, it works well with a virtualenv, and all I had to do was:

    mkvirtualenv pydev
    curl -o install_superpack.sh https://raw.github.com/fonnesbeck/ScipySuperpack/master/install_superpack.sh
    sh install_superpack.sh

This installed most of what I need, and then I installed Sympy and [IPython][] as well:

    pip install sympy
    pip install ipython[zmq,qtconsole,notebook,test]

[Anaconda]: https://store.continuum.io/
[Enthought Canopy]: https://www.enthought.com/products/canopy/
[ScipySuperpack]: http://fonnesbeck.github.io/ScipySuperpack/
[IPython]: http://ipython.org/ipython-doc/stable/install/install.html

### Statsmodels

Statsmodels has a git repository, so I first cloned that into my own Git account, and got a local copy:

    git clone git@github.com:ChadFulton/statsmodels.git

ScipySuperpack comes with statsmodels=0.5.0, so I need to remove that and start using my local copy, which I can do via ```python setup.py develop```

    pip uninstall statsmodels
    cd statsmodels
    python setup.py develop

Now, ```pip freeze``` shows the local development copy of statsmodels:

    -e git+git@github.com:ChadFulton/statsmodels.git@661c4dc86b2c4c9fc4ce7c63efcded94b90ed0de#egg=statsmodels-dev

### Code Standards

Finally, there are two packages not installed above that will be helpful for developing: [pep8] and [pylint]. pep8 checks code for adherence to the [pep8 python style guide][], and pylint is a static code checker.

Installing these is as easy as

    pip install pep8 pylint

[pep8]: https://pypi.python.org/pypi/pep8
[pylint]: https://pypi.python.org/pypi/pylint
[pep8 python style guide]: http://www.python.org/dev/peps/pep-0008/
