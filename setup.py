from setuptools import find_packages, setup

setup(
    name="local_connectivity",
    version='0.1',
    author='Andi Bergeson',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
          'numpy',
          'pandas',
          'matplotlib',
          'statsmodels',
          'tqdm',
          'seaborn',
          'caveclient'
          ]
)