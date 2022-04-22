from abc import ABC, abstractmethod
from typing import List

from reactea.chem.compounds import Compound
from reactea.optimization.problem import Problem
from reactea.utils.constatns import EAConstants


class AbstractEA(ABC):
    """"""

    def __init__(self,
                 problem: Problem,
                 initial_population: List[Compound],
                 max_generations: int = EAConstants.MAX_GENERATIONS,
                 mp: bool = True,
                 visualizer: bool = False):
        """"""
        self.initial_population = initial_population
        self.problem = problem
        self.initial_population = initial_population
        self.max_generations = max_generations
        self.mp = mp
        self.visualizer = visualizer

    def run(self):
        """"""
        if self.problem.fevaluation is None or len(self.problem.fevaluation) == 0:
            raise ValueError("At leat one objective should be provided.")

        if self.problem.number_of_objectives == 1:
            final_pop = self._run_so()
        else:
            final_pop = self._run_mo()

        return final_pop

    @abstractmethod
    def _run_so(self):
        """"""
        raise NotImplementedError

    @abstractmethod
    def _run_mo(self):
        """"""
        raise NotImplementedError
