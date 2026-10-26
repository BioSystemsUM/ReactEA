# ReactEA: Combining Evolutionary Algorithms with Reaction Rules Towards Focused Molecular Design

[![PyPI](https://img.shields.io/pypi/v/reactea)](https://pypi.org/project/reactea/)
[![Python versions](https://img.shields.io/pypi/pyversions/reactea)](https://pypi.org/project/reactea/)
[![Documentation](https://img.shields.io/readthedocs/reactea)](https://reactea.readthedocs.io)
[![License](https://img.shields.io/github/license/BioSystemsUM/ReactEA)](LICENSE)

**Documentation: [reactea.readthedocs.io](https://reactea.readthedocs.io)**

### Description

ReactEA is a reaction-based single and multi-objective evolutionary approach towards focused molecular design. ReactEA is a 
modular and problem-agnostic method that uses enzymatic reaction rules to manipulate molecules.
The generated molecules are optimized for user-specified objective functions using a vast suite of EAs implemented in 
the [jMetalPy framework](https://github.com/jMetal/jMetalPy).

### Table of contents:

- [Requirements](#requirements)
- [Installation](#installation)
    - [Pip](#pip)
    - [From GitHub](#from-github)
- [Development](#development)
- [Getting Started](#getting-started)
    - [Using ReactEA](#using-reactea)
    - [Results](#results)
    - [Plotting the transformation routes](#plotting-the-transformation-routes)
<!-- - [About Us](#about-us) -->
- [Citing ReactEA](#citing-reactea)
- [License](#licensing)


## Requirements

- Python >= 3.11

The dependencies (RDKit, jMetalPy, NumPy, pandas, ChEMBL Structure Pipeline, ...) are declared in
[pyproject.toml](pyproject.toml) and installed automatically.

## Installation

### Pip

Install ReactEA via pip:

```bash
pip install reactea
```

or add it to a [uv](https://docs.astral.sh/uv/) project:

```bash
uv add reactea
```

### From GitHub

Alternatively, install ReactEA from source. The project is managed with [uv](https://docs.astral.sh/uv/)
(see the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)).

1. Clone the repository:
```bash
git clone https://github.com/BioSystemsUM/ReactEA.git
cd ReactEA
```

2. Create the virtual environment (`.venv`) and install ReactEA with the pinned dependencies from `uv.lock`:
```bash
uv sync
```
uv downloads a compatible Python if needed. To choose a specific version use, e.g., `uv sync --python 3.13`.

Without uv, `pip install .` also works.

## Development

`uv sync` also installs the development tools (pytest, flake8, mypy).

```bash
uv run pytest            # run the tests
uv run flake8 src tests  # lint
```

To add or upgrade a dependency use `uv add <package>` or `uv lock --upgrade-package <package>`,
and commit the updated `uv.lock`.

To build the documentation locally:

```bash
uv run --group docs sphinx-build docs docs/_build/html
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

n_rings_feval = evaluation_functions_wrapper(number_of_rings, 
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
minimize_rings = case_study_wrapper(n_rings_feval, 
                                    multi_objective=False, 
                                    name='minimize_rings')

# SINGLE-OBJECTIVE CASE STUDY WITH MULTIPLE EVALUATION FUNCTIONS
# case study to optimize a single objective but with multiple evaluation functions `f1` and `f2` (minimize number of rings in a molecule and maximize qed)
# the number of evaluation functions must be the same as the number of values in weights and the sum of the weights must be 1
minimize_rings_maximize_qed = case_study_wrapper([n_rings_feval, qed_feval], 
                                                 multi_objective=False, 
                                                 name='minimize_rings_maximize_qed', 
                                                 weights=[0.3, 0.7])

# MULTI-OBJECTIVE CASE STUDY
# case study to optimize multiple objectives simultaneous
minimize_rings_maximize_qed_mo = case_study_wrapper([n_rings_feval, qed_feval], 
                                                    multi_objective=True, 
                                                    name='minimize_rings_maximize_qed_mo')
```

Evaluation functions run serially by default. For expensive ones (e.g. docking), evaluate the population in
parallel with `n_jobs` (`-1` uses all cores):

```python
expensive_feval = evaluation_functions_wrapper(my_expensive_function, maximize=True, worst_fitness=0.0,
                                               name='expensive', n_jobs=-1)
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

# Maximum number of (random) reaction rules tried in each mutation until one produces a valid product (maximum is 22949)
max_rules_by_iter: 22949
# **Mutant selected will be randomly chosen from the compounds with similarity between `best_similarity` and `best_similarity - tolerance`
tolerance: 0.1

# Number of generations to run the algorithm
generations: 100
# EA to use (see below)
algorithm: "NSGAIII"

# Path to output folder
output_path: ".../output_dir_path/"

# Whether to draw the transformation route of each improved solution (SVG files in the `routes` folder of the output)
save_routes: True
```

Algorithms:

- Single-objective case studies: `GA`, `ES`, `SA` and `LS`, as well as the multi-objective algorithms below.
- Multi-objective case studies: `NSGAII`, `NSGAIII`, `SPEA2`, `IBEA` and `RandomSearch`.
- `SA`, `LS` and `RandomSearch` start from a single compound (`init_pop_size: 1`).

Optional parameters (defaults in parentheses):

```yaml
# stop if the mean fitness does not improve for this number of generations (off)
patience: 10
# probability of applying the mutation and crossover operators (1.0)
mutation_probability: 1.0
crossover_probability: 1.0
# show the Pareto front approximation in a live plot, multi-objective only (False)
visualize: False

# SA: initial and minimum temperature and cooling rate (1.0, 0.000001, 0.95)
temperature: 1.0
minimum_temperature: 0.000001
alpha: 0.95
# ES: keep the parents in the next generation (True)
elitist: True
# IBEA: fitness scaling factor (1.0)
kappa: 1.0
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

case_study_rings = minimize_rings_maximize_qed_mo
# provide path to configuration file and case study
run_reactea(configs_path = 'config.yaml', 
            case_study = case_study_rings)
```

`configs_path` can also be a dictionary with the configurations.

To stop the run with other criteria (e.g. when a solution reaches a score), pass a `termination_criterion`
(see [Termination criteria](https://reactea.readthedocs.io/en/latest/customization.html#termination-criteria)).

### Results

The results are saved in `<output_path>/<exp_name>/<algorithm>/`:

| File | Content |
|---|---|
| `FINAL_<time>.csv` | final population: SMILES and the score of each objective |
| `FINAL_<time>_UNIQUE_SOLUTIONS.csv` | the same, without repeated solutions |
| `FINAL_TRANSFORMATIONS_<time>.csv` | route of each solution: intermediate compounds and reaction rule ids |
| `ReactionMutationLogs.txt` | every transformation applied during the run (parent, product, rule id and the parent's scores) |
| `configs.yaml` | configurations used in the run |
| `routes/` | drawings of the transformation routes (if `save_routes: True`) |

The csv files use `;` as separator.
Scores are saved as returned by the evaluation functions, both for maximized and minimized objectives.

### Plotting the transformation routes

Each solution's route (seed compound → intermediates → final compound) can be drawn as a reaction scheme.
Each step's arrow shows the reaction rule's EC numbers, co-reactants and id, the atoms changed by each reaction are
highlighted, and the objective values are shown below the final compound.

```python
from reactea.visualization import load_routes, draw_route, save_routes

# routes of the final population (output folder of the run or a FINAL_TRANSFORMATIONS_*.csv file)
routes = load_routes('output_dir_path/NSGAIII_EXAMPLE_CONFIG/NSGAIII')

drawing = draw_route(routes[0])  # displayed inline in Jupyter notebooks
drawing.save('route_0.svg')      # vector (for papers); use .png for a raster image

# save all improved routes (one file per solution)
save_routes(routes, 'routes/')
```

`draw_route` options include `ncols` (molecules per row), `panel_size`, `font_size`, `align`, `highlight_changes`,
`show_rule_info` and `show_scores` (see [Results](https://reactea.readthedocs.io/en/latest/results.html)).

See the [documentation](https://reactea.readthedocs.io) for more details: all the configuration options, how to
implement your own evaluation functions, case studies and termination criteria, and the API reference.

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

Accepted at [GECCO 2023](https://gecco-2023.sigevo.org/HomePage).

Publication:

João Correia, Vítor Pereira, and Miguel Rocha. 2023. Combining Evolutionary Algorithms with Reaction Rules Towards Focused Molecular Design. In Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '23). Association for Computing Machinery, New York, NY, USA, 900–909. https://doi.org/10.1145/3583131.3590413

Bibtex entry:

```markdown
@inproceedings{Correia2023,
  doi = {10.1145/3583131.3590413},
  url = {https://doi.org/10.1145/3583131.3590413},
  year = {2023},
  month = jul,
  publisher = {{ACM}},
  author = {Jo{\~{a}}o Correia and V{\'{\i}}tor Pereira and Miguel Rocha},
  title = {Combining Evolutionary Algorithms with Reaction Rules Towards Focused Molecular Design},
  booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference}
}
``` 

## Licensing

Reactea is under [GPL-3.0 license](LICENSE).
