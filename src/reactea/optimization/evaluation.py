from abc import ABC, abstractmethod
from functools import reduce
from typing import List, Union

import numpy as np
from rdkit.Chem import MolFromSmarts, Mol, MolToSmiles
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.QED import qed

from reactea.utilities.io import Loaders


class ChemicalEvaluationFunction(ABC):
    """
    Base class for chemical evaluation functions.
    Child classes must implement the get_fitness and method_str methods.
    """

    def __init__(self, maximize: bool = True, worst_fitness: float = -1.0):
        """
        Initializes the Chemical Evaluation Function class.

        Parameters
        ----------
        maximize: bool
            If it is a maximization problem.
        worst_fitness: float
            The worst fitness that can given to a solution.
        """
        self.maximize = maximize
        self.worst_fitness = worst_fitness

    @abstractmethod
    def get_fitness(self, candidates: Union[Mol, List[Mol]]):
        """
        Get fitness of a set of solutions.

        Parameters
        ----------
        candidates: Union[Mol, List[Mol]]
            Mol objects to get fitness from.
        Returns
        -------
        Union[List[Num], List[List[Num]]
            Fitness of each Mol object
        """
        raise NotImplementedError

    @abstractmethod
    def method_str(self):
        """
        Get name of the evaluation function.

        Returns
        -------
        str:
            name of the evaluation function.
        """
        raise NotImplementedError

    def __str__(self):
        return self.method_str()

    def __call__(self, candidate: Union[Mol, List[Mol]]):
        return self.get_fitness(candidate)


class DummyEvalFunction(ChemicalEvaluationFunction):
    """
    Dummy evaluation function.
    """

    def __init__(self, maximize: bool = True, worst_fitness: float = -1.0):
        """
        Initializes a Dummy Evaluation Function.

        Parameters
        ----------
        maximize: bool
            if the goal is to maximize (True) or minimize (False) the fitness of the evaluation function.
        worst_fitness: float
            The worst fitness possible for the evaluation function.
        """
        super(DummyEvalFunction, self).__init__(maximize, worst_fitness)

    def get_fitness(self, candidates: Union[Mol, List[Mol]]):
        """
        Returns the fitness of a set of Mol objects.
        In this case it's only the size of the molecule' SMILES string.

        Parameters
        ----------
        candidates: Union[Mol, List[Mol]]
            Mol objects to evaluate.

        Returns
        -------
        List[int]:
            weighted fitness of the evaluation functions.
        """
        if isinstance(candidates, list):
            return [len(MolToSmiles(candidate)) for candidate in candidates]
        else:
            return [len(MolToSmiles(candidates))]

    def method_str(self):
        """
        Get name of the evaluation function.

        Returns
        -------
        str:
            name of the evaluation function
        """
        return "DummyEF"


class AggregatedSum(ChemicalEvaluationFunction):
    """
    AggregatedSum evaluation function.
    Transforms multiple chemical evaluation functions into a single weighted fitness score.
    Useful to use  in case we want to use multiple evaluation functions in single objective problems.
    """

    def __init__(self,
                 fevaluation: List[ChemicalEvaluationFunction],
                 tradeoffs: List[float] = None,
                 maximize: bool = True,
                 worst_fitness: float = 0.0):
        """
        Initializes a AggregatedSum Evaluation Function.

        Parameters
        ----------
        fevaluation: List[ChemicalEvaluationFunction]
            list of chemical evaluation functions.
        tradeoffs: List[float]
            list of tradeoffs/weights between the evaluation functions.
        maximize: bool
            if the goal is to maximize (True) or minimize (False) the fitness of the evaluation function.
        worst_fitness: float
            The worst fitness possible for the evaluation function.
        """
        super(AggregatedSum, self).__init__(maximize, worst_fitness)
        self.fevaluation = fevaluation
        if tradeoffs and len(tradeoffs) == len(fevaluation):
            self.tradeoffs = tradeoffs
        else:
            self.tradeoffs = [1 / len(self.fevaluation)] * (len(self.fevaluation))

    def get_fitness(self, candidates: Union[Mol, List[Mol]]):
        """
        Returns the fitness of a set of Mol objects.
        In this case it's the weighted fitness of multiple evaluation functions.

        Parameters
        ----------
        candidates: Union[Mol, List[Mol]]
            Mol objects to evaluate.

        Returns
        -------
        List[int]:
            length of the Mol objects' SMILES strings .
        """
        if isinstance(candidates, list):
            evals = []
            for f in self.fevaluation:
                evals.append(f.get_fitness(candidates))
            res = np.transpose(evals)
            return np.dot(res, self.tradeoffs)
        else:
            evals = []
            for f in self.fevaluation:
                evals.append(f.get_fitness(candidates))
            res = np.transpose(evals)
            return np.dot(res, self.tradeoffs)

    def method_str(self):
        """
        Get the names of the evaluation functions.

        Returns
        -------
        str:
            name of the evaluation functions.
        """
        return "Aggregated Sum = " + reduce(lambda a, b: a + " " + b, [f.method_str() for f in self.fevaluation], "")


class SweetnessPredictionDeepSweet(ChemicalEvaluationFunction):
    """
    Sweetness Prediction using the tool DeepSweet.
    For more info see: https://github.com/BioSystemsUM/DeepSweet
    """

    def __init__(self, maximize=True, worst_fitness=-1.0):
        """
        Initializes the Sweetness Prediction DeepSweet evaluation function.

        Parameters
        ----------
        maximize: bool
            if the goal is to maximize (True) or minimize (False) the fitness of the evaluation function.
        worst_fitness: float
            The worst fitness possible for the evaluation function.
        """
        super(SweetnessPredictionDeepSweet, self).__init__(maximize, worst_fitness)
        self.ensemble = Loaders.load_deepsweet_ensemble()

    def _predict_sweet_prob(self, mol: Mol):
        """
        Internal method to predict sweetness probability using the ensemble from DeepSweet.

        Parameters
        ----------
        mol: Mol
            Mol object to predict sweetness from.

        Returns
        -------
        List[int]
            list with the sweetness prediction.
        """
        try:
            res, _ = self.ensemble.predict([mol])
        except Exception:
            res = [self.worst_fitness]
        return res

    def get_fitness(self, candidates: Union[Mol, List[Mol]]):
        """
        Returns the fitness of a set of Mol objects.
        In this case it's the sweetness probability of the molecules.

        Parameters
        ----------
        candidates: Union[Mol, List[Mol]]
            Mol objects to evaluate.

        Returns
        -------
        List[int]:
            list of sweetness probability of the candidate Mol objects.
        """
        if isinstance(candidates, list):
            scores = []
            for mol in candidates:
                scores.extend(self._predict_sweet_prob(mol))
            return scores
        else:
            return self._predict_sweet_prob(candidates)

    def method_str(self):
        """
        Get name of the evaluation function.

        Returns
        -------
        str:
            name of the evaluation function
        """
        return "Sweetness Prediction (DeepSweet)"


class Caloric(ChemicalEvaluationFunction):
    """
    Caloric evaluation function.
    Penalizes molecules with specific groups that are related with being caloric or not.
    """

    def __init__(self, maximize: bool = True, worst_fitness: float = -1.0):
        """
        Initializes the Caloric evaluation function.

        Parameters
        ----------
        maximize: bool
            if the goal is to maximize (True) or minimize (False) the fitness of the evaluation function.
        worst_fitness: float
            The worst fitness possible for the evaluation function.
        """
        super(Caloric, self).__init__(maximize=maximize, worst_fitness=worst_fitness)

    def _match_score(self, candidate: Mol):
        """
        Internal method to identify how many groups that match the SMARTS "[Or5,Or6,Or7,Or8,Or9,Or10,Or11,Or12]" are
        present in a molecule. Molecules with more matches are more penalized.

        Parameters
        ----------
        candidate: Mol
            Mol object to evaluate.

        Returns
        -------
        List[float]
            caloric score
        """
        try:
            caloric_smarts = MolFromSmarts("[Or5,Or6,Or7,Or8,Or9,Or10,Or11,Or12]")
            n_matches = len(candidate.GetSubstructMatches(caloric_smarts))
            if n_matches > 0:
                return [self.worst_fitness]
            else:
                return [1.0]
        except Exception:
            return [self.worst_fitness]

    def get_fitness(self, candidates: Union[Mol, List[Mol]]):
        """
        Returns the fitness of a set of Mol objects.
        In this case it's the caloric score of the molecules.

        Parameters
        ----------
        candidates: Union[Mol, List[Mol]]
            Mol objects to evaluate.

        Returns
        -------
        List[int]:
            list of caloric scores of the candidate Mol objects.
        """
        if isinstance(candidates, list):
            scores = []
            for mol in candidates:
                scores.extend(self._match_score(mol))
            return scores
        else:
            return self._match_score(candidates)

    def method_str(self):
        """
        Get name of the evaluation function.

        Returns
        -------
        str:
            name of the evaluation function
        """
        return "Caloric"


class LogP(ChemicalEvaluationFunction):
    """
    LogP evaluation function.
    Computes the partition coefficient.
    """

    def __init__(self, maximize=True, worst_fitness=-100.0):
        """
        Initializes the LogP evaluation function.

        Parameters
        ----------
        maximize: bool
            if the goal is to maximize (True) or minimize (False) the fitness of the evaluation function.
        worst_fitness: float
            The worst fitness possible for the evaluation function.
        """
        super(LogP, self).__init__(maximize=maximize, worst_fitness=worst_fitness)

    def _get_logp(self, mol: Mol):
        """
        Computes the partition coefficient of a molecule.

        Parameters
        ----------
        mol: Mol
            Mol object to calculate the partition coefficient

        Returns
        -------
        List[float]
            list with the partition coefficient of the molecule
        """
        try:
            score = MolLogP(mol)
        except Exception:
            score = self.worst_fitness
        return [score]

    def get_fitness(self, candidates: Union[Mol, List[Mol]]):
        """
        Returns the fitness of a set of Mol objects.
        In this case it's the partition coefficient of the molecules.

        Parameters
        ----------
        candidates: Union[Mol, List[Mol]]
            Mol objects to evaluate.

        Returns
        -------
        List[int]:
            list of partition coefficients of the candidate Mol objects.
        """
        if isinstance(candidates, list):
            scores = []
            for mol in candidates:
                scores.extend(self._get_logp(mol))
            return scores
        else:
            return self._get_logp(candidates)

    def method_str(self):
        """
        Get name of the evaluation function.

        Returns
        -------
        str:
            name of the evaluation function
        """
        return "logP"


class QED(ChemicalEvaluationFunction):
    """
    QED evaluation function.
    Computes the drug-likeliness of a molecule based on the similarity of the distributions of a set of properties
    with known drugs.
    """

    def __init__(self, maximize=True, worst_fitness=-10.0):
        """
        Initializes the QED evaluation function.

        Parameters
        ----------
        maximize: bool
            if the goal is to maximize (True) or minimize (False) the fitness of the evaluation function.
        worst_fitness: float
            The worst fitness possible for the evaluation function.
        """
        super(QED, self).__init__(maximize, worst_fitness)

    def _qed(self, mol: Mol):
        """
        Computes the drug-likeliness of a molecule.

        Parameters
        ----------
        mol: Mol
            Mol object to calculate the drug-likeliness

        Returns
        -------
        List[float]
            list with the drug-likeliness score of the molecule
        """
        try:
            score = qed(mol)
        except Exception:
            score = self.worst_fitness
        return [score]

    def get_fitness(self, candidates: Union[Mol, List[Mol]]):
        """
        Returns the fitness of a set of Mol objects.
        In this case it's the drug-likeliness score of the molecules.

        Parameters
        ----------
        candidates: Union[Mol, List[Mol]]
            Mol objects to evaluate.

        Returns
        -------
        List[int]:
            list of drug-likeliness scores of the candidate Mol objects.
        """
        if isinstance(candidates, list):
            scores = []
            for mol in candidates:
                scores.extend(self._qed(mol))
            return scores
        else:
            return self._qed(candidates)

    def method_str(self):
        """
        Get name of the evaluation function.

        Returns
        -------
        str:
            name of the evaluation function
        """
        return "QED"
