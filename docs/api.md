# API reference

## Running ReactEA

```{eval-rst}
.. autofunction:: reactea.run_reactea

.. autoclass:: reactea.ChemicalEA
   :members: run
```

## Case studies

```{eval-rst}
.. autofunction:: reactea.case_study_wrapper

.. autoclass:: reactea.case_studies.CaseStudy

.. autoclass:: reactea.case_studies.CompoundQuality

.. autoclass:: reactea.case_studies.SweetReactor

.. autoclass:: reactea.optimization.problem.ChemicalProblem
```

## Evaluation functions

```{eval-rst}
.. autofunction:: reactea.evaluation_functions_wrapper

.. automodule:: reactea.optimization.evaluation
   :exclude-members: get_fitness_single, method_str
```

## Visualization

```{eval-rst}
.. automodule:: reactea.visualization.routes
```

## Termination criteria

```{eval-rst}
.. automodule:: reactea.optimization.jmetal.terminators
```

## Chemistry

```{eval-rst}
.. autoclass:: reactea.chem.Compound

.. autoclass:: reactea.chem.ReactionRule

.. autoclass:: reactea.chem.ChemUtils

.. autoclass:: reactea.chem.ChEMBLStandardizer
```

## Input and output

```{eval-rst}
.. autoclass:: reactea.io_streams.Loaders

.. autoclass:: reactea.io_streams.Writers
```