# Getting started

A ReactEA run needs three things:

1. the **evaluation functions** that score the molecules,
2. a **case study** that combines them into the optimization problem,
3. a **configuration** with the seed compounds and the algorithm to use.

## 1. Evaluation functions

An evaluation function scores an RDKit `Mol`.
Any function can be wrapped with {func}`~reactea.evaluation_functions_wrapper`:

```python
from rdkit.Chem.QED import qed
from reactea import evaluation_functions_wrapper


def number_of_rings(mol):
    return len(mol.GetRingInfo().AtomRings())


n_rings = evaluation_functions_wrapper(number_of_rings,
                                       maximize=False,    # minimize the number of rings
                                       worst_fitness=100, # score if the function fails
                                       name='n_rings')

qed_score = evaluation_functions_wrapper(qed, maximize=True, worst_fitness=0.0, name='qed')
```

ReactEA also has [built-in evaluation functions](api.md#evaluation-functions)
(molecular weight, logP, QED, synthetic accessibility, similarity, docking, ...).

## 2. Case study

A case study defines whether the evaluation functions are optimized as separate objectives (multi-objective) or as
a weighted sum (single-objective):

```python
from reactea import case_study_wrapper

# multi-objective: one objective per evaluation function
rings_and_qed = case_study_wrapper([n_rings, qed_score], multi_objective=True, name='rings_and_qed')

# single-objective: weighted sum of the evaluation functions (the weights must sum to 1)
rings_and_qed_so = case_study_wrapper([n_rings, qed_score], multi_objective=False, name='rings_and_qed_so',
                                      weights=[0.3, 0.7])

# single-objective with one evaluation function
rings = case_study_wrapper(n_rings, multi_objective=False, name='rings')
```

## 3. Configuration

The configuration is a YAML file (or a dictionary):

```yaml
exp_name: "MY_EXPERIMENT"

# seed compounds: tsv file with a "smiles" column (and optionally a "compound_id" column)
init_pop_path: "seeds.tsv"
init_pop_size: 20

max_rules_by_iter: 22949
tolerance: 0.1

generations: 100
algorithm: "NSGAII"

output_path: "outputs/"
save_routes: True
```

See [Configuration](configuration.md) for all the options.

## 4. Run

```python
from reactea import run_reactea

run_reactea('config.yaml', rings_and_qed)
```

The results are saved in `outputs/MY_EXPERIMENT/NSGAII/` (see [Results](results.md)).

## Command line

The built-in case studies (`CompoundQuality` and `SweetReactor`) can also be run from the command line, with the
case study set in the configuration file:

```yaml
case_study: "CompoundQuality"
multi_objective: True
```

```bash
reactea config.yaml
```

## Examples

The [examples folder](https://github.com/BioSystemsUM/ReactEA/tree/main/examples) has a configuration file for
each algorithm and Jupyter notebooks:

- [example.ipynb](https://github.com/BioSystemsUM/ReactEA/blob/main/examples/example.ipynb): a complete run, with
  the routes plotted at the end.
- [evaluation_functions.ipynb](https://github.com/BioSystemsUM/ReactEA/blob/main/examples/implementation_examples/evaluation_functions.ipynb),
  [case_studies.ipynb](https://github.com/BioSystemsUM/ReactEA/blob/main/examples/implementation_examples/case_studies.ipynb)
  and [terminators.ipynb](https://github.com/BioSystemsUM/ReactEA/blob/main/examples/implementation_examples/terminators.ipynb):
  how to implement your own components.