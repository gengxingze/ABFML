from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
    name="abfml",
    version='1.0.0',
    description='Rapid building, fitting, and application of machine learning force fields',
    author='GengXingZe',
    author_email='1308186024@qq.com',
    python_requires='>=3.11',
    install_requires=['ase==3.22', 'numpy==1.25', 'matplotlib>=3.8', 'torch>=2.0', 'jsonschema>=4.23.0'],
    packages=['abfml',
              'abfml/data',
              'abfml/entrypoints',
              'abfml/logger',
              'abfml/loss',
              'abfml/model',
              'abfml/optimizer',
              'abfml/param',
              'abfml/train',
              'abfml/utils'
              ],

    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
    entry_points={'console_scripts': ['abfml = abfml.main:main']}

)
