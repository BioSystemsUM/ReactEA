# Customization

## Evaluation functions

{func}`~reactea.evaluation_functions_wrapper` covers most cases (see [Getting started](getting_started.md)).
For more control, subclass {class}`~reactea.optimization.evaluation.ChemicalEvaluationFunction` and implement
`get_fitness_single` (score of one molecule) and `method_str` (name of the function):

```python
from rdkit.Chem import Mol
from rdkit.Chem.Descriptors import MolWt

from reactea.optimization.evaluation import ChemicalEvaluationFunction


class MolecularWeightInRange(ChemicalEvaluationFunction):
    """1.0 if the molecular weight is in the range, 0.0 otherwise."""

    def __init__(self, min_weight: float = 200, max_weight: float = 500, maximize: bool = True,
                 worst_fitness: float = 0.0, n_jobs: int = 1):
        super().__init__(maximize, worst_fitness, n_jobs)
        self.min_weight = min_weight
        self.max_weight = max_weight

    def get_fitness_single(self, candidate: Mol):
        try:
            return float(self.min_weight <= MolWt(candidate) <= self.max_weight)
        except Exception:
            # invalid molecule
            return self.worst_fitness

    def method_str(self):
        return 'MolecularWeightInRange'
```

`get_fitness` evaluates a list of molecules (in parallel if `n_jobs` is not `1`). Override it if your function
is faster on a whole batch at once (e.g. a machine learning model).

## Case studies

{func}`~reactea.case_study_wrapper` covers most cases. For more control, subclass
{class}`~reactea.case_studies.CaseStudy` and implement `objective` (returns the
{class}`~reactea.optimization.problem.ChemicalProblem` with the evaluation functions), `name` and `feval_names`
(names of the objectives, used as the columns of the results file):

```python
from reactea.case_studies import CaseStudy
from reactea.optimization.evaluation import AggregatedSum, QED, SAS
from reactea.optimization.problem import ChemicalProblem


class DrugLikeAndSynthesizable(CaseStudy):

    def __init__(self, multi_objective: bool = True):
        super().__init__(multi_objective)

    def objective(self):
        if self.multi_objective:
            return ChemicalProblem([QED(), SAS()])
        return ChemicalProblem([AggregatedSum([QED(), SAS()], [0.5, 0.5])])

    def name(self):
        return 'DrugLikeAndSynthesizable'

    def feval_names(self):
        # one name per objective, separated by ";"
        return 'QED;SAS' if self.multi_objective else '0.5*QED + 0.5*SAS'
```

## Termination criteria

By default, a run stops after `generations` generations, or earlier if `patience` is set and the solutions stop
improving (see [Configuration](configuration.md)).

Other criteria are passed to {func}`~reactea.run_reactea` with `termination_criterion`.
ReactEA has these built-in criteria (in {mod}`reactea.optimization.jmetal.terminators`):

| Criterion | Stops when |
|---|---|
| `StoppingByEvaluations(max_generations)` | the maximum number of generations is reached |
| `StoppingByEvaluationsOrImprovement(patience)` | the mean score does not improve, in any objective, for `patience` generations |
| `StoppingByEvaluationsOrMeanFitnessValue(expected_value)` | the mean score reaches `expected_value` in every objective |
| `StoppingByScore(value, objective=0)` | a solution reaches `value` in an objective |

```python
from reactea import run_reactea
from reactea.optimization.jmetal.terminators import StoppingByScore

# stop when a solution reaches a QED of 0.9 (the first objective), or after `generations` generations
run_reactea('config.yaml', DrugLikeAndSynthesizable(), termination_criterion=StoppingByScore(0.9, objective=0))
```

All criteria also stop after a maximum number of generations: `max_generations` if given, otherwise `generations`
from the configuration. A criterion keeps the state of a run (e.g. the number of generations), so create a new one
for each run.

### Writing your own

Subclass {class}`~reactea.optimization.jmetal.terminators.ReactEATerminationCriterion` and implement `stop`. It is
called at each generation with the scores of the current solutions (one list per solution, one value per objective,
as returned by the evaluation functions) and whether each objective is maximized, and returns True to stop the run.
Counting the generations and the maximum number of generations are handled by the base class:

```python
from reactea import run_reactea
from reactea.optimization.jmetal.terminators import ReactEATerminationCriterion


class StopWhenEnoughGoodSolutions(ReactEATerminationCriterion):
    """Stops when n solutions reach a minimum score in every (maximized) objective."""

    def __init__(self, n: int, min_score: float, max_generations: int = None):
        super().__init__(max_generations)
        self.n = n
        self.min_score = min_score

    def stop(self, scores, maximize):
        good = [s for s in scores if all(value >= self.min_score for value in s)]
        return len(good) >= self.n


run_reactea('config.yaml', DrugLikeAndSynthesizable(), termination_criterion=StopWhenEnoughGoodSolutions(3, 0.8))
```
