# Results

The results of a run are saved in `<output_path>/<exp_name>/<algorithm>/`:

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
In single-objective runs with several evaluation functions, the column is the weighted sum
(e.g. `0.3*n_rings + 0.7*qed`).

```python
import pandas as pd

final = pd.read_csv('outputs/MY_EXPERIMENT/NSGAII/FINAL_10-25_09-05-20.csv', sep=';')
```

## Plotting the transformation routes

Each solution's route (seed compound → intermediates → final compound) can be drawn as a reaction scheme:

- the atoms changed by each reaction are highlighted,
- each arrow shows the reaction rule's EC numbers, co-reactants and id,
- the scores are shown below the final compound.

```python
from reactea.visualization import load_routes, draw_route, save_routes

# routes of the final population (the run's output folder or a FINAL_TRANSFORMATIONS_*.csv file)
routes = load_routes('outputs/MY_EXPERIMENT/NSGAII')

route = max(routes, key=lambda r: r.n_steps)
drawing = draw_route(route)   # displayed inline in Jupyter notebooks
drawing.save('route.svg')     # vector image (for papers)
drawing.save('route.png')     # raster image

# save the routes of all improved solutions (one file per solution)
save_routes(routes, 'routes/')
```

```{image} _static/route_example.svg
:alt: Example of a transformation route
```

{func}`~reactea.visualization.routes.draw_route` options:

| Option | Default | Description |
|---|---|---|
| `ncols` | `4` | molecules per row (longer routes wrap to the next row) |
| `panel_size` | `(300, 260)` | size of each molecule, in pixels |
| `arrow_width` | `170` | width of the arrows, in pixels |
| `font_size` | `13` | font size of the labels |
| `align` | `True` | draw each molecule aligned with the previous one |
| `highlight_changes` | `True` | highlight the atoms changed by each reaction |
| `show_rule_info` | `True` | show the EC numbers, co-reactants and rule id on the arrows |
| `show_scores` | `True` | show the scores below the final compound |

{func}`~reactea.visualization.routes.save_routes` skips the solutions that were not changed (`skip_unchanged=True`) and
repeated routes (`unique=True`), and saves SVG files by default (`fmt='png'` for PNG).