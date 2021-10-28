from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name='diffusion_source',
    version='1.0',
    description='package for computing confidence sets on diffusion source procedures',
    license="MIT",
    long_description=long_description,
    author='Quinn Dawkins',
    author_email='qed4wg@virginia.edu',
    url="https://github.com/lab-sigma/Diffusion-Source-Identification",
    packages=['diffusion_source'],
    install_requires=['networkx', 'numpy', 'scipy', 'matplotlib', 'pandas'],
)
