# ReactEA: Combining Evolutionary Algorithms with Reaction Rules Towards Focused Molecular Design

### Description

ReactEA is a reaction-based single and multi-objective evolutionary approach towards focused molecular design. ReactEA is a 
modular and problem-agnostic method that uses enzymatic reaction rules to manipulate molecules.
The generated molecules are optimized for user-specified objective functions using a vast suite of EAs implemented in 
the [jMetalPy framework](https://github.com/jMetal/jMetalPy).

### Table of contents:

- [Requirements](#requirements)
- [Installation](#installation)
    - [Pip](#pip)
    - [From Github](#from-github)
- [Getting Started](#getting-started)
    - [Using ReactEA](#using-reactea)
<!-- - [About Us](#about-us) -->
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
- chembl_structure_pipeline
- joblib==1.1.0
- networkx==2.6.3
- click==8.1.3
  

## Installation

### Pip

Install DeepMol via pip or conda:

```bash
pip install git+https://github.com/BioSystemsUM/ReactEA.git
```

### From GitHub

Alternatively, install dependencies and ReactEA manually.

1. Clone the repository:
```bash
git clone https://github.com/BioSystemsUM/ReactEA.git
```

3. Install dependencies:
```bash
python setup.py install
```

## Getting Started

### Using ReactEA:

- Define the evaluation functions (case study) to use in the optimization 
(see [evaluation_functions.ipynb](examples/implementation_examples/evaluation_functions.ipynb) 
and [case_studies.ipynb](examples/implementation_examples/case_studies.ipynb) for more details).

Example:

```python
from rdkit.Chem.QED import qed
from reactea import evaluation_functions_wrapper

# EVALUATION FUNCTIONS

# evaluation function returning the number of rings a molecule
def number_of_rings(mol):
    ri = mol.GetRingInfo()
    n_rings = len(ri.AtomRings())
    return n_rings

n_rigs_feval = evaluation_functions_wrapper(number_of_rings, 
                                            maximize=False, 
                                            worst_fitness=100, 
                                            name='n_rings')

# evaluation function returning the drug-likeliness score (QED) of a molecule
def qed_score(mol):
    return qed(mol)

qed_feval = evaluation_functions_wrapper(qed_score, 
                                         maximize=True, 
                                         worst_fitness=0.0, 
                                         name='qed')

# CASE STUDY

from reactea import case_study_wrapper

# SINGLE OBJECTIVE CASE STUDY
# case study to optimize a single objective `f1` (minimize number of rings in a molecule)
minimize_rings = case_study_wrapper(n_rigs_feval, 
                                    multi_objective=False, 
                                    name='minimize_rings')

# SINGLE-OBJECTIVE CASE STUDY WITH MULTIPLE EVALUATION FUNCTIONS
# case study to optimize a single objective but with multiple evaluation functions `f1` and `f2` (minimize number of rings in a molecule and maximize qed)
# the number of evaluation functions must be the same as the number of values in weights and the sum of the weights must be 1
minimize_rings_maximize_qed = case_study_wrapper([n_rigs_feval, qed_feval], 
                                                 multi_objective=False, 
                                                 name='minimize_rings_maximize_qed', 
                                                 weights=[0.3, 0.7])

# MULTI-OBJECTIVE CASE STUDY
# case study to optimize multiple objectives simultaneous
minimize_rings_maximize_qed_mo = case_study_wrapper([n_rigs_feval, qed_feval], 
                                                    multi_objective=True, 
                                                    name='minimize_rings_maximize_qed_mo')
```
- Provide the configuration file (see configuration files in [config_files](examples/config_files/) for more details).

Example:

```yaml
# CONFIGURATION FILE

# Name of the experiment (results will be saved in a folder with this name (inside output folder))
exp_name: "NSGAIII_EXAMPLE_CONFIG"

# Path to the file containing the seed compounds (with column named smiles)
init_pop_path: ".../.../path_to_seed_compounds.tsv"
# size of the initial population to sample from the seed compounds (if not provided, all seed compounds will be used)
init_pop_size: 100
# whether to standardize the seed compounds (if not provided, the seed compounds will not be standardized)
standardize: True

# Maximum number of reaction rules to try in each generation (maximum is 22949)
max_rules_by_iter: 22949
# **Mutant selected will be randomly chosen from the compounds with similarity between `best_similarity` and `best_similarity - tolerance`
tolerance: 0.1

# Number of generations to run the algorithm
generations: 100
# EA to use (NSGAIII, NSGAII, SPEA2, IBEA, GA, LS, ES and SA)
algorithm: "NSGAIII"

# Path to output folder
output_path: ".../output_dir_path/"
```

Note: ** When generating reaction products (offspring) from the molecules from the previous generation (parents), 
multiple products will be generated for each parent, including cofactors like water, carbon dioxide,
acetylCoA, etc. These cofactors have no interest in the optimization process, to eliminate them and select compounds
with relevance we only select from a pool of compounds that are similar to the parent compound. This pool is formed by
the compounds within a range of similarity (tolerance). The range is defined by the similarity of the most 
similar compound (`best_similarity`) and `best_similarity - tolerance`.

- Run ReactEA:

```python
from reactea import run_reactea

case_study_rings = minimize_rings_maximize_qed_mo()
# provide path to configuration file and case study
run_reactea(configs_path = 'config.yaml', 
            case_study = case_study_rings)
```

<!--
### Using the Command Line Interface

```bash
reactea config_file_path
```
-->

<!--
## About Us

ReactEA is managed by a team of contributors from the BioSystems group 
at the Centre of Biological Engineering, University of Minho.

This research was financed by Portuguese Funds through FCT – Fundação para 
a Ciência e a Tecnologia.
-->

## Citing ReactEA

Manuscript under preparation.

## Licensing

Reactea is under [GPL-3.0 license](LICENSE).