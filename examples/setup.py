from distutils.core import setup
from Cython.Build import cythonize

""" Build Script for Inquiry Examples. Run 'python setup.py build_ext --inplace'  """

setup(
    ext_modules = cythonize(['network.pyx', 'utilities.pyx'])
    )
                                 
    
