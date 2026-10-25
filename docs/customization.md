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

Other criteria subclass jMetalPy's `TerminationCriterion` and implement `update` (called at each generation with
the current solutions) and `is_met`. To use one, build the {class}`~reactea.ChemicalEA` directly:

```python
from jmetal.util.termination_criterion import TerminationCriterion

from reactea import ChemicalEA, Loaders, Writers


class StopWhenQEDReached(TerminationCriterion):
    """Stops when a solution reaches a QED score (the first objective) or after max_generations."""

    def __init__(self, qed: float, max_generations: int):
        super().__init__()
        self.qed = qed
        self.max_generations = max_generations
        self.generations = 0
        self.best = 0.0

    def update(self, *args, **kwargs):
        self.generations += 1
        # jMetalPy minimizes, so maximized objectives are negated
        self.best = max(-s.objectives[0] for s in kwargs['SOLUTIONS'])

    @property
    def is_met(self):
        return self.best >= self.qed or self.generations >= self.max_generations


configs = Loaders.get_config_from_yaml('config.yaml')
configs['output_dir'] = configs['output_dir'] / configs['algorithm']
Writers.set_up_folders(configs['output_dir'])
init_pop, init_pop_smiles = Loaders.initialize_population(configs)

case_study = DrugLikeAndSynthesizable()
ea = ChemicalEA(configs['algorithm'], case_study.objective(), initial_population=init_pop,
                reaction_rules=Loaders.initialize_rules(), max_generations=configs['generations'], configs=configs)
ea.termination_criterion = StopWhenQEDReached(0.9, configs['generations'])
final_population = ea.run()

Writers.save_final_pop(final_population, configs, case_study.feval_names())
Writers.save_intermediate_transformations(final_population, configs)
```