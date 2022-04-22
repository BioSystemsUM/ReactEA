from abc import ABC, abstractmethod
from functools import reduce
from typing import List, Union

import numpy as np
from rdkit import DataStructs
from rdkit.Chem import MolFromSmarts, Mol, rdMolDescriptors
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.QED import qed

from reactea.utils.chem_utils import ChemUtils
from reactea.utils.io import Loaders


class ChemicalEvaluationFunction(ABC):
    """"""

    def __init__(self, maximize: bool = True, worst_fitness: float = -1.0):
        """"""
        self.maximize = maximize
        self.worst_fitness = worst_fitness

    def get_fitness(self, candidate: Mol, batched: bool):
        """"""
        return self._get_fitness_batch(candidate) if batched else self._get_fitness_single(candidate)

    @abstractmethod
    def _get_fitness_single(self, candidate: Mol):
        """"""
        return NotImplementedError

    @abstractmethod
    def _get_fitness_batch(self, list_candidates: List[Mol]):
        """"""
        return NotImplementedError

    @abstractmethod
    def method_str(self):
        """"""
        return NotImplementedError

    def __str__(self):
        """"""
        return self.method_str()

    def __call__(self, candidate: Union[Mol, List[Mol]], batched: bool):
        """"""
        return self.get_fitness(candidate, batched)


class DummyEvalFunction(ChemicalEvaluationFunction):
    """"""

    def __init__(self, maximize: bool = True, worst_fitness: float = -1.0):
        """"""
        super(DummyEvalFunction, self).__init__(maximize, worst_fitness)

    def _get_fitness_single(self, candidate: Mol):
        """"""
        return len(candidate)

    def _get_fitness_batch(self, list_candidates: List[Mol]):
        """"""
        return [len(candidate) for candidate in list_candidates]

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

    def _get_fitness_single(self, candidate):
        """"""
        res = []
        for f in self.fevaluation:
            res.append(f._get_fitness_single(candidate))
        return np.dot(res, self.tradeoffs)

    def _get_fitness_batch(self, list_mols):
        """"""
        evals = []
        for f in self.fevaluation:
            evals.append(f._get_fitness_batch(list_mols))
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

    def _get_fitness_single(self, candidate: Mol):
        """"""
        raise NotImplementedError

    def _get_fitness_batch(self, list_candidates: List[Mol]):
        """"""
        raise NotImplementedError

    def method_str(self):
        return "Sweetness Prediction (DeepSweet)"


class SweetnessPrediction(ChemicalEvaluationFunction):
    """"""

    def __init__(self, configs, maximize=True, worst_fitness=0.0):
        """"""
        super(SweetnessPrediction, self).__init__(maximize, worst_fitness)
        self.config = configs
        self.m1, self.m2, self.m3 = Loaders.loadSweetModels(self.config)

    def _get_fitness_single(self, candidate):
        """"""
        if candidate:
            fps = np.zeros((1,))
            desc = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(candidate, 2048)
            DataStructs.ConvertToNumpyArray(desc, fps)
            fps = np.expand_dims(fps, axis=0)
            score1 = self.m1.predict_proba(fps)[0, 1]
            score2 = self.m2.predict_proba(fps)[0, 1]
            score3 = self.m3.predict_proba(fps)[0, 0]
            res = score1 + score2 + score3
            res = res / 3
        else:
            res = self.worst_fitness
        return res

    def _get_fitness_batch(self, list_mols):
        """"""
        atomPairsArray, invalids = ChemUtils.atomPairsDescriptors(list_mols)
        scores = np.zeros((len(list_mols), 3))
        scores[:, 0] = self.m1.predict_proba(atomPairsArray)[:, 1]
        scores[:, 1] = self.m2.predict_proba(atomPairsArray)[:, 1]
        scores[:, 2] = self.m3.predict(atomPairsArray)[:, 0]
        scores = np.sum(scores, axis=1) / 3.0
        scores[invalids] = self.worst_fitness
        return scores

    def method_str(self):
        return "Sweetness Prediction"


class Caloric(ChemicalEvaluationFunction):
    """"""

    def __init__(self, maximize=True, worst_fitness=0):
        """"""
        super(Caloric, self).__init__(maximize=maximize, worst_fitness=worst_fitness)

    def _get_fitness_single(self, candidate: Mol):
        """"""
        if candidate:
            caloric_smarts = MolFromSmarts("[Or5,Or6,Or7,Or8,Or9,Or10,Or11,Or12]")
            n_matchs = len(candidate.GetSubstructMatches(caloric_smarts))
            score = 1 / (n_matchs*2 + 1)
        else:
            score = self.worst_fitness
        return score

    def _get_fitness_batch(self, list_mols: List[Mol]):
        """"""
        listScore = []
        for mol in list_mols:
            if mol:
                caloric_smarts = MolFromSmarts("[Or5,Or6,Or7,Or8,Or9,Or10,Or11,Or12]")
                n_matchs = len(mol.GetSubstructMatches(caloric_smarts))
                score = 1 / (n_matchs*2 + 1)
                listScore.append(score)
            else:
                listScore.append(self.worst_fitness)
        return listScore

    def method_str(self):
        return "Caloric"


class LogP(ChemicalEvaluationFunction):
    """"""

    def __init__(self, maximize=True, worst_fitness=-100.0):
        super(LogP, self).__init__(maximize=maximize, worst_fitness=worst_fitness)

    def _get_fitness_single(self, candidate: Mol):
        """"""
        if candidate:
            score = MolLogP(candidate)
        else:
            score = self.worst_fitness
        return score

    def _get_fitness_batch(self, list_mols: List[Mol]):
        """"""
        list_scores = []
        for mol in list_mols:
            try:
                list_scores.append(MolLogP(mol))
            except Exception:
                list_scores.append(self.worst_fitness)
        return list_scores

    def method_str(self):
        return "logP"


class QED(ChemicalEvaluationFunction):
    """"""

    def __init__(self, maximize=True, worst_fitness=-10.0):
        super(QED, self).__init__(maximize, worst_fitness)

    def _get_fitness_single(self, candidate: Mol):
        """"""
        if candidate:
            score = qed(candidate)
        else:
            score = self.worst_fitness
        return score

    def _get_fitness_batch(self, list_mols):
        """"""
        list_scores = []
        for mol in list_mols:
            try:
                list_scores.append(qed(mol))
            except Exception:
                list_scores.append(self.worst_fitness)
        return list_scores

    def method_str(self):
        return "QED"
