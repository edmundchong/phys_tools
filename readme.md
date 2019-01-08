# phystools


This package can be used to preprocess, load and visualize electrophysiology data.

## Installation:

Anaconda Python distribution works best! Please install the version with Python >= 3.5.

You will need to install the following packages through conda. Most are included with the 
distribution (in __bold__ are not installed by default as far as I remember).

1. Numpy
2. PyQt5
3. __tqdm__
4. PyTables
5. __numba__

Before installing, please make sure that Anaconda's executable is your default Python interpreter
(`which python` should point to some kind of anaconda folder)

Once this is done, just run `python setup.py install`

## Next steps

* _For preprocessing data_, check out the readme in the "[preprocess](phys_tools/preprocess/)" folder
* _For analysis_, check out the readme in the "[models](phys_tools/models/)" folder'

In the "[examples](examples/)" folder, there are a couple notebooks that document how to use this too.

There is also a couple rudamentary guis for visualizing data, but it is relatively undocumented.
They can be run from the shell as `odorNavigator` for odor data or `patterNavigator` for optogenetic
recordings. If you have questions, let me know!

