# ReactEA

ReactEA is a reaction-based single and multi-objective evolutionary approach towards focused molecular design.
It uses enzymatic reaction rules to transform molecules, so every compound it generates comes with the route of
reactions that leads to it from a seed compound.

The generated molecules are optimized for user-defined objectives using the evolutionary algorithms of
[jMetalPy](https://github.com/jMetal/jMetalPy).

```{image} _static/route_example.svg
:alt: Example of a transformation route found by ReactEA
```

ReactEA is:

- **Problem-agnostic:** any function that scores an RDKit molecule can be an objective.
- **Single or multi-objective:** GA, ES, SA and LS, or NSGA-II, NSGA-III, SPEA2, IBEA and Random Search.
- **Explainable:** each solution is reached through enzymatic reaction rules, which can be drawn as reaction schemes.

```{toctree}
:maxdepth: 2
:caption: User guide

installation
getting_started
configuration
results
customization
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
changelog
citing
```