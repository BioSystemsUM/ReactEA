from abc import ABC, abstractmethod
from functools import reduce
from typing import List, Union

import numpy as np
from rdkit.Chem import MolFromSmarts, Mol
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.QED import qed

from reactea.utilities.io import Loaders


class ChemicalEvaluationFunction(ABC):
    """"""

    def __init__(self, maximize: bool = True, worst_fitness: float = -1.0):
        """"""
        self.maximize = maximize
        self.worst_fitness = worst_fitness

    @abstractmethod
    def get_fitness(self, candidates: Union[Mol, List[Mol]]):
        """"""
        raise NotImplementedError

    @abstractmethod
    def method_str(self):
        """"""
        raise NotImplementedError

    def __str__(self):
        """"""
        return self.method_str()

    def __call__(self, candidate: Union[Mol, List[Mol]]):
        """"""
        return self.get_fitness(candidate)


class DummyEvalFunction(ChemicalEvaluationFunction):
    """"""

    def __init__(self, maximize: bool = True, worst_fitness: float = -1.0):
        """"""
        super(DummyEvalFunction, self).__init__(maximize, worst_fitness)

    def get_fitness(self, candidates: Union[Mol, List[Mol]]):
        """"""
        if isinstance(candidates, list):
            return [len(candidate) for candidate in candidates]
        else:
            return len(candidates)

    def method_str(self):
        """"""
        return "DummyEF"


class AggregatedSum(ChemicalEvaluationFunction):
    """"""

    def __init__(self,
                 fevaluation: List[ChemicalEvaluationFunction],
                 tradeoffs: List[float] = None,
                 maximize: bool = True,
                 worst_fitness: float = 0.0):
        """"""
        super(AggregatedSum, self).__init__(maximize, worst_fitness)
        self.fevaluation = fevaluation
        if tradeoffs and len(tradeoffs) == len(fevaluation):
            self.tradeoffs = np.array(tradeoffs)
        else:
            self.tradeoffs = np.array([1 / len(self.fevaluation)] * (len(self.fevaluation)))

    def get_fitness(self, candidates):
        """"""
        if isinstance(candidates, list):
            res = []
            for f in self.fevaluation:
                res.append(f.get_fitness(candidates))
            return np.dot(res, self.tradeoffs)
        else:
            evals = []
            for f in self.fevaluation:
                evals.append(f.get_fitness(candidates))
            evals = np.transpose(np.array(evals))
            res = np.dot(evals, self.tradeoffs)
            return res

    def method_str(self):
        return "Aggregated Sum = " + reduce(lambda a, b: a + " " + b, [f.method_str() for f in self.fevaluation], "")


class SweetnessPredictionDeepSweet(ChemicalEvaluationFunction):
    """"""

    def __init__(self, configs: dict, maximize=True, worst_fitness=-1.0):
        super(SweetnessPredictionDeepSweet, self).__init__(maximize, worst_fitness)
        self.configs = configs
        self.ensemble = Loaders.load_deepsweet_ensemble()

    def _predict_sweet_prob(self, mol: Mol):
        try:
            res, _ = self.ensemble.predict([mol])
        except Exception:
            res = [self.worst_fitness]
        return res

    def get_fitness(self, candidates: Union[Mol, List[Mol]]):
        """"""
        if isinstance(candidates, list):
            scores = []
            for mol in candidates:
                scores.extend(self._predict_sweet_prob(mol))
            return scores
        else:
            return self._predict_sweet_prob(candidates)

    def method_str(self):
        return "Sweetness Prediction (DeepSweet)"


class Caloric(ChemicalEvaluationFunction):
    """"""

    def __init__(self, maximize=True, worst_fitness=0):
        """"""
        super(Caloric, self).__init__(maximize=maximize, worst_fitness=worst_fitness)

    def _match_score(self, candidate: Mol, penalty_ratio: int = 3):
        try:
            caloric_smarts = MolFromSmarts("[Or5,Or6,Or7,Or8,Or9,Or10,Or11,Or12]")
            n_matches = len(candidate.GetSubstructMatches(caloric_smarts))
            score = 1 / (n_matches * penalty_ratio + 1)
        except Exception:
            score = self.worst_fitness
        return score

    def get_fitness(self, candidates: Union[Mol, List[Mol]]):
        if isinstance(candidates, list):
            scores = []
            for mol in candidates:
                scores.append(self._match_score(mol))
            return scores
        else:
            return self._match_score(candidates)

    def method_str(self):
        return "Caloric"


class LogP(ChemicalEvaluationFunction):
    """"""

    def __init__(self, maximize=True, worst_fitness=-100.0):
        super(LogP, self).__init__(maximize=maximize, worst_fitness=worst_fitness)

    def _get_logp(self, mol: Mol):
        try:
            score = MolLogP(mol)
        except Exception:
            score = self.worst_fitness
        return score

    def get_fitness(self, candidates: Union[Mol, List[Mol]]):
        if isinstance(candidates, list):
            scores = []
            for mol in candidates:
                scores.append(self._get_logp(mol))
            return scores
        else:
            return self._get_logp(candidates)

    def method_str(self):
        return "logP"


class QED(ChemicalEvaluationFunction):
    """"""

    def __init__(self, maximize=True, worst_fitness=-10.0):
        super(QED, self).__init__(maximize, worst_fitness)

    def _qed(self, mol: Mol):
        try:
            score = qed(mol)
        except Exception:
            score = self.worst_fitness
        return score

    def get_fitness(self, candidates: Union[Mol, List[Mol]]):
        if isinstance(candidates, list):
            scores = []
            for mol in candidates:
                scores.append(self._qed(mol))
            return scores
        else:
            return self._qed(candidates)

    def method_str(self):
        return "QED"
