from setuptools import setup, find_packages

setup(
    name='polaris',
    version='0.5.0',
    description='Reinforcement Learning library based on sonnet and ray tune.',
    author='Victor Villin',
    author_email='victor.villin@unine.ch',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
    "numpy>=1.26.4,<=2.0.0",
    "dm_tree",
    "ray>=2.9.0",
    "ray[tune]",
    "gymnasium==1.2.0",
    "dm-sonnet==2.0.2",
    "ml_collections",
    "tensorflow-probability==0.24.0",
    "sacred",
    "wandb",
    "plotly",
    "scipy",
    "tqdm",
    ],
    extras_require={
        "gpu": ["tensorflow[and-cuda]==2.18.0", "tf_keras==2.18.0"]
    }
)

