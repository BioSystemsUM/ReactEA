# Changelog

## 2.0.1 (unreleased)

- New `termination_criterion` argument in `run_reactea` and `ChemicalEA` to stop the run with any criterion.
- New `ReactEATerminationCriterion` base class: a new criterion only implements `stop(scores, maximize)`, with the
  scores as returned by the evaluation functions; generations and the maximum number of generations are handled by
  the base class.
- New `StoppingByScore` criterion: stops when a solution reaches a score.
- The built-in criteria use the base class. Their `max_evaluations` argument is renamed `max_generations` and is
  optional (`generations` from the configuration is used if not given).
  `StoppingByEvaluationsOrMeanFitnessValue` compared jMetalPy's internal objectives instead of the scores, so it
  never stopped for maximized objectives and stopped in the wrong direction for minimized ones; it also counted
  evaluations instead of generations. `StoppingByEvaluationsOrImprovement` counted the first generation as one
  without improvement.
- `FINAL_<time>_UNIQUE_SOLUTIONS.csv` now uses `;` as separator, like `FINAL_<time>.csv` (it used `,`).
- Documentation on Read the Docs.

## 2.0.0

Modernised ReactEA: Python 3.11+, current dependencies, a new route plotting module, and many bug fixes.

### Breaking changes

- **Python 3.11+ required** (3.11–3.14 supported). Dependencies upgraded to numpy 2, pandas 3, RDKit 2026.9
  (`rdkit-pypi` → `rdkit`), jmetalpy 1.9.
- **Plotting API replaced:** `reactea.vizualization.PlotResults` is removed; use {mod}`reactea.visualization.routes`.
  `ChemUtils.smiles_to_img`, `ChemUtils.mol_to_image` and `Loaders.load_results_case` are removed.
- **Results files changed:** minimised objectives are now saved with their real sign (they were negated), in both
  `FINAL_*.csv` and `ReactionMutationLogs.txt`. Single-objective runs with several functions now have one correctly
  named column (e.g. `0.3*n_rings + 0.7*qed`). Paths in `configs.yaml` are saved as plain strings.
- **SweetReactor single-objective objective changed:** it now includes `SimilarityToInitial` with the intended
  weights. Before, the weights were silently replaced by equal weights.
- **RandomSearch changed:** it is now a random walk with reaction rules. Before, it only returned the seed.
- **Evaluation runs serially by default.** For expensive functions, pass `n_jobs=-1`.

### New

- Route plotting: draw each solution's route as a reaction scheme (see [Results](results.md)).
- Project managed with uv; `run_reactea` accepts a config `dict` and respects `visualize`.

### Fixes

- Out-of-memory crashes: reaction rules are compiled the first time they're used (rule loading went from 2.8 GB to
  374 MB).
- Single-objective SweetReactor crashed with numpy ≥ 1.24.
- Non-elitist ES (`elitist: False`) never evolved.
- Reactions with more than 1,000 matches only ever modified the first atoms of the molecule; the products are now
  sampled across the whole molecule.
- Live Pareto plot (`visualize`) showed the worst solutions; also fixed for matplotlib 3.11.
- The CLI no longer requires the `verbose` and `visualize` keys.
- `SimilarityToInitial` crashed on invalid molecules (numpy 2).
- IBEA compatibility with jmetalpy 1.9.

### Performance

- Python 3.14 is no longer ~5× slower (joblib `loky` backend).
- `PenalizedSweetness` loads the DeepSweet models once instead of on every evaluation.
- `SAS` no longer slows down over long runs.

## 1.0.0

First release.