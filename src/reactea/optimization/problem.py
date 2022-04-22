from abc import ABC, abstractmethod
from typing import List, Union

from reactea.chem.compounds import Compound
from reactea.optimization.evaluation import ChemicalEvaluationFunction


class Problem(ABC):
    """"""

    def __init__(self, name: str, fevaluation: List[ChemicalEvaluationFunction]):
        """"""
        self.name = name
        if fevaluation is None:
            raise ValueError("At least one evaluation function needs to be provided")
        else:
            self.fevaluation = fevaluation
        self.number_of_objectives = len(self.fevaluation)

    @property
    def is_maximization(self):
        """"""
        return all([f.maximize for f in self.fevaluation])

    def __str__(self):
        """"""
        return '{0} ({1} objectives)'.format(self.__class__.__name__, self.number_of_objectives)

    def __repr__(self):
        """"""
        return self.__class__.__name__

    def evaluate_solution(self, candidates: Union[Compound, List[Compound]], batched: bool):
        """"""
        return self._evaluate_solution_batch(candidates) if batched else self._evaluate_solution_single(candidates)

    @abstractmethod
    def _evaluate_solution_batch(self, candidates):
        """"""
        raise NotImplementedError

    @abstractmethod
    def _evaluate_solution_single(self, candidates):
        """"""
        raise NotImplementedError


class ChemicalProblem(Problem):
    """"""

    def __init__(self, fevaluation: List[ChemicalEvaluationFunction], configs: dict):
        """"""
        super(ChemicalProblem, self).__init__("ChemicalProblem", fevaluation)
        self.configs = configs

    def _evaluate_solution_batch(self, candidates: List[Compound]):
        """"""
        list_mols = [smi.mol for smi in candidates]
        evals = []
        for f in self.fevaluation:
            evals.append(f(list_mols, batched=True))
        return list(zip(*evals))

    def _evaluate_solution_single(self, candidates: Compound):
        """"""
        candidates = candidates.mol
        evals = []
        for f in self.fevaluation:
            evals.append(f(candidates, batched=False))
        return evals

    def get_name(self):
        """"""
        return "ChemicalProblem"

