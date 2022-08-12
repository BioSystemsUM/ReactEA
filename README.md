# ReactEA: Combining Evolutionary Algorithms with Reaction Rules Towards Focused Molecular Design

**[README UNDER CONSTRUCTION!]**

### Description

...

### Table of contents:

- [Requirements](#requirements)
- [Installation](#installation)
    - [Pip](#pip)
    - [Docker](#docker)
- [Getting Started](#getting-started)
    - [Using ReactEA](#using-reactea)
- [About Us](#about-us)
- [Citing ReactEA](#citing-reactea)
- [License](#licensing)


## Requirements

- rdkit-pypi==2022.03.1
- numpy==1.21.5
- pandas==1.3.5
- cytoolz==0.11.2
- jmetalpy
- PyYAML==6.0
- matplotlib==3.5.1
  

## Installation

### Pip

Install DeepMol via pip or conda:

```bash
pip install reactea #just for example (not working)
```

or

```bash
conda install -c conda-forge reactea #just for example (not working)
```

### Docker

(IN PREPARATION - NOT FUNCTIONAL YET!)
1. Install [docker](https://docs.docker.com/install/).
2. Pull an existing image (X.XGb to download) from DockerHub:

```bash
docker pull XXX
```

or clone the repository and build it manually:

```bash
git clone https://github.com/BioSystemsUM/ReactEA.git
docker build ...
```

3. Create a container:
```bash
docker run ...
```

### Manually

(IN PREPARATION - NOT FUNCTIONAL YET!)

Alternatively, install dependencies and ReactEA manually.

1. Clone the repository:
```bash
git clone https://github.com/BioSystemsUM/DeepMol.git
```

3. Install dependencies:
```bash
python setup.py install
```

## Getting Started

### Using ReactEA:

- Define a seed population of initial molecules.

- Define the reaction rules to use (SMARTS), possible coreactants and the number of maximum rules to use per mutation.

- Define the evaluation functions to use in the optimization.

- Select the algorithm to use.

- Define the number of generations.


## About Us

ReactEA is managed by a team of contributors from the BioSystems group 
at the Centre of Biological Engineering, University of Minho.

This research was financed by Portuguese Funds through FCT – Fundação para 
a Ciência e a Tecnologia.

## Citing ReactEA

Manuscript under preparation.

## Licensing

Reactea is under [GPL-3.0 license](https://raw.githubusercontent.com/BioSystemsUM/ReactEA/main/LICENSE?token=GHSAT0AAAAAABRR6Q6KOOQLKSYY3CL2BU66YXJHEXA).