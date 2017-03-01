# from distutils.core import setup
from setuptools import setup

setup(
    name='phys_tools',
    version='',
    packages=['phys_tools', 'phys_tools.utils', 'phys_tools.preprocess', 'phys_tools.models'],
    url='',
    license='MIT',
    author='chris',
    author_email='cdw291@nyu.edu',
    description='', install_requires=[],
    entry_points={'console_scripts': [
        'patterNavigator = phys_tools.patterNavigator:main',
        'odorNavigator = phys_tools.odorNavigator:main']}
)
