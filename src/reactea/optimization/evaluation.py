from abc import ABC, abstractmethod
from typing import List, Union


class EvaluationFunction(ABC):
    """"""

    def __init__(self, maximize: bool = True, worst_fitness: float = -1.0):
        """"""
        self.maximize = maximize
        self.worst_fitness = worst_fitness

    def get_fitness(self, candidate: str, batched: bool):
        """"""
        return self._get_fitness_batch(candidate) if batched else self._get_fitness_single(candidate)

    @abstractmethod
    def _get_fitness_single(self, candidate: str):
        """"""
        return NotImplementedError

    @abstractmethod
    def _get_fitness_batch(self, list_candidates: List[str]):
        """"""
        return NotImplementedError

    @abstractmethod
    def method_str(self):
        """"""
        return NotImplementedError

    def __str__(self):
        """"""
        return self.method_str()

    def __call__(self, candidate: Union[str, List[str]], batched: bool):
        """"""
        return self.get_fitness(candidate, batched)


class DummyEvalFunction(EvaluationFunction):
    """"""

    def __init__(self, maximize: bool = True, worst_fitness: float = -1.0):
        """"""
        super(DummyEvalFunction, self).__init__(maximize, worst_fitness)

    def _get_fitness_single(self, candidate: str):
        """"""
        return len(candidate)

    def _get_fitness_batch(self, list_candidates: List[str]):
        """"""
        return [len(candidate) for candidate in list_candidates]

    def method_str(self):
        """"""
        return "DummyEF"
