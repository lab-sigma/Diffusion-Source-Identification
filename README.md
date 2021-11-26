# Diffusion Source Identification  on Networks with Statistical Confidence

Source code for confidence set analysis on the Diffusion Source Identification (DSI) problem. Paper presented at ICML 2021 can be found at [ICML 2021's Paper List](https://proceedings.mlr.press/v139/dawkins21a.html) or on [arXiv](https://arxiv.org/abs/2106.04800).

This is an updated version of the code. Previous version can be found in the legacy directory.

## Installation

The code can be installed by running the following commands

```
    git clone https://github.com/lab-sigma/Diffusion-Source-Identification
    cd Diffusion-Source-Identification
    pip install .
```

The package is named ``diffusion\_source''. The relevant subpackages are summarized below.

### diffusion\_source.graphs

Provides a wrapper around a networkx graph used by the infection model. A superset of the synthetic networks used in the paper are provided, as well as classes for importing a graph from adjacency formats and an already created networkx graph.

### diffusion\_source.infection\_model

A base abstract infection model class is provided as ``InfectionModelBase'' that includes a definition of the confidence set function for arbitrary diffusion processes.

### diffusion\_source.discrepancies

A number of discrepancy functions are written for use when running the confidence set algorithm

### diffusion\_source.display

A set of tools for displaying example infected sets and confidence sets, as well as generating summary figures for large scale model tests.
