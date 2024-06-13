from setuptools import setup, find_packages

setup(
    name='polaris',
    version='0.1.0',
    description='Policy Gradient Lib based on sonnet and ray tune.',
    author='Victor Villin',
    author_email='victor.villin@unine.ch',
    packages=find_packages(exclude=('tests', 'docs'))
)