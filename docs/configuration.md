# Configuration

The configuration can be a YAML file or a dictionary (both can be passed to {func}`~reactea.run_reactea`).

## Required

| Option | Description |
|---|---|
| `exp_name` | name of the experiment (results are saved in `<output_path>/<exp_name>/<algorithm>/`) |
| `init_pop_path` | tsv file with the seed compounds (`smiles` column and, optionally, `compound_id`) |
| `max_rules_by_iter` | maximum number of random reaction rules tried in each mutation until one produces a valid product (there are 22949 rules) |
| `tolerance` | similarity tolerance used to choose the product of a reaction (see below) |
| `generations` | number of generations |
| `algorithm` | algorithm to use (see [Algorithms](#algorithms)) |
| `output_path` | folder where the results are saved |

## Optional

| Option | Default | Description |
|---|---|---|
| `init_pop_size` | all seeds | number of seed compounds randomly sampled from `init_pop_path` |
| `standardize` | `False` | standardize the seed compounds ([ChEMBL Structure Pipeline](https://github.com/chembl/ChEMBL_Structure_Pipeline)) |
| `save_routes` | `False` | draw the route of each improved solution (SVG files in the `routes` folder) |
| `patience` | off | stop if the mean fitness does not improve for this number of generations |
| `mutation_probability` | `1.0` | probability of applying the mutation operator |
| `crossover_probability` | `1.0` | probability of applying the crossover operator |
| `visualize` | `False` | show the Pareto front approximation in a live plot (multi-objective only) |
| `temperature` | `1.0` | SA: initial temperature |
| `minimum_temperature` | `0.000001` | SA: minimum temperature |
| `alpha` | `0.95` | SA: cooling rate |
| `elitist` | `True` | ES: keep the parents in the next generation |
| `kappa` | `1.0` | IBEA: fitness scaling factor |

Command line only:

| Option | Description |
|---|---|
| `case_study` | built-in case study (`CompoundQuality` or `SweetReactor`) |
| `multi_objective` | optimize the case study as multi-objective (`True`) or single-objective (`False`) |
| `verbose` | show the RDKit logs (default `False`) |

## Algorithms

| Case study | Algorithms |
|---|---|
| single-objective | `GA`, `ES`, `SA`, `LS`, `NSGAII`, `NSGAIII`, `SPEA2` and `IBEA` |
| multi-objective | `NSGAII`, `NSGAIII`, `SPEA2`, `IBEA` and `RandomSearch` |

`SA`, `LS` and `RandomSearch` evolve a single compound, so they need `init_pop_size: 1`.

`RandomSearch` is a random walk: at each generation a random reaction rule is applied to the previous compound,
without selection, and the non-dominated compounds found are kept.

## How mutations work

Every algorithm uses the same operator to create new compounds: random reaction rules are tried on the compound
(up to `max_rules_by_iter`) until one produces a valid product.

A reaction usually produces several compounds, including co-products like water, CO₂ or acetyl-CoA.
To keep the compound of interest, the new compound is chosen at random among the products whose similarity to the
original compound is between `best_similarity - tolerance` and `best_similarity`, where `best_similarity` is the
similarity of the most similar product.

## Parallel evaluation

Evaluation functions run serially by default, which is fastest for cheap functions.
For expensive ones (e.g. docking), evaluate the population in parallel with `n_jobs` (`-1` uses all cores):

```python
from reactea import evaluation_functions_wrapper
from reactea.optimization.evaluation import Docking

expensive = evaluation_functions_wrapper(my_expensive_function, maximize=True, worst_fitness=0.0,
                                         name='expensive', n_jobs=-1)
docking = Docking('DRD2', n_jobs=-1)
```