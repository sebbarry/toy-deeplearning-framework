import setuptools
from setuptools import find_packages


with open("README.md", 'r', encoding="utf-8") as f: 
    long_description = f.read()

setuptools.setup(
        name='toy_dl', 
        version='0.0.1', 
        author='Sebastian Barry', 
        author_email='sebastianbarry02@gmail.com', 
        long_description=long_description, 
        description='Toy Deep Learning Framework',
        url='https://github.com/sebbarry/toy-deeplearning-framework.git', 
        license='MIT', 
        install_requires=['numpy'],
        packages=['toy_dl']
        )


