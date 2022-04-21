from typing import List

from reactea.optimization.ea import AbstractEA
from reactea.optimization.problem import Problem
from reactea.utils.constatns import EAConstants


class EA(AbstractEA):
    """"""

    def __init__(self,
                 problem: Problem,
                 initial_population: List[str] = None,
                 max_generations: int = EAConstants.MAX_GENERATIONS,
                 mp: bool = True,
                 visualizer: bool = False,
                 algorithm: str = None,
                 batched: bool = True,
                 configs: dict = None
                 ):
        super(EA, self).__init__(problem, initial_population, max_generations, mp, visualizer)
        pass

    def _convertPopulation(self, population: List):
        """"""
        raise NotImplementedError

    def _run_so(self):
        """"""
        raise NotImplementedError

    def _run_mo(self):
        """"""
        raise NotImplementedError