from abc import ABC, abstractmethod
from typing import List

from reactea.chem.compounds import Compound
from reactea.optimization.problem import Problem


class AbstractEA(ABC):
    """
    Base class for EAs.
    Child classes must implement the _run_so and _sun_mo methods.
    """

    def __init__(self,
                 problem: Problem,
                 initial_population: List[Compound],
                 max_generations: int = 10,
                 visualizer: bool = False):
        """
        Initializes a EA.

        Parameters
        ----------
        problem: Problem
            Problem to use in the EA.
        initial_population: List[Compound]
            List of Compound objects to use as initial population.
        max_generations: int
            number of max generations
        visualizer: bool
            use a visualizer (True) or not (False)

        """
        self.initial_population = initial_population
        self.problem = problem
        self.initial_population = initial_population
        self.max_generations = max_generations
        self.visualizer = visualizer

    def run(self):
        """
        Run EA.

        Returns
        -------
        List[Compound]:
            List of final solutions (Compound objects).
        """
        if self.problem.fevaluation is None or len(self.problem.fevaluation) == 0:
            raise ValueError("At least one objective should be provided.")

        if self.problem.number_of_objectives == 1:
            final_pop = self._run_so()
        else:
            final_pop = self._run_mo()

        return final_pop

    @abstractmethod
    def _run_so(self):
        """
        Run single-objective EA.

        Returns
        -------
        List[Compound]:
            List of final solutions (Compound objects).
        """
        raise NotImplementedError

    @abstractmethod
    def _run_mo(self):
        """
        Run multi-objective EA.

        Returns
        -------
        List[Compound]:
            List of final solutions (Compound objects).
        """
        raise NotImplementedError
