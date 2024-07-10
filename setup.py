from setuptools import setup, find_packages

setup(
    name='polaris',
    version='0.2.0',
    description='Reinforcement Learning library based on sonnet and ray tune.',
    author='Victor Villin',
    author_email='victor.villin@unine.ch',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
    "dm_tree",
    "ray==2.9.0",
    "ray[tune]",
    "gymnasium>=0.27.1",
    "tensorflow>=2.12.0",
    "dm-sonnet==2.0.2",
    "ml_collections",
    "sacred",
    ]
)