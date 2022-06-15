from abc import ABC, abstractmethod
from functools import reduce
from typing import List, Union

import numpy as np
from rdkit import DataStructs
from rdkit.Chem import MolFromSmarts, Mol, MolToSmiles, GetSymmSSSR, EnumerateStereoisomers, AllChem, MolFromSmiles
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.EnumerateStereoisomers import StereoEnumerationOptions
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
                 worst_fitness: str = 'mean'):
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
        worst_fitness: str
            Function to compute worst fitness from the passed ChemicalEvaluationFunctions (choose from mean, max, min).
        """
        if worst_fitness == 'mean':
            self.worst_fitness = np.mean([f.worst_fitness for f in fevaluation])
        elif worst_fitness == 'max':
            self.worst_fitness = np.max([f.worst_fitness for f in fevaluation])
        else:
            self.worst_fitness = np.min([f.worst_fitness for f in fevaluation])
        super(AggregatedSum, self).__init__(maximize, self.worst_fitness)
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

    def __init__(self, maximize=True, worst_fitness=0.0):
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
            return self.ensemble.predict([mol])[0]
        except:
            return [self.worst_fitness]

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


class PenalizedSweetness(ChemicalEvaluationFunction):
    """
    Sweetness Prediction using the tool DeepSweet with caloric penalty.
    For more info see: https://github.com/BioSystemsUM/DeepSweet
    """

    def __init__(self, maximize=True, worst_fitness=0.0):
        """
        Initializes the Penalized Sweetness evaluation function.

        Parameters
        ----------
        maximize: bool
            if the goal is to maximize (True) or minimize (False) the fitness of the evaluation function.
        worst_fitness: float
            The worst fitness possible for the evaluation function.
        """
        super(PenalizedSweetness, self).__init__(maximize, worst_fitness)
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
        float
            sweetness probability.
        """
        try:
            return self.ensemble.predict([mol])[0][0]
        except:
            return self.worst_fitness

    def _match_score(self, mol: Mol):
        """
        Internal method to identify how many groups that match the SMARTS "[Or5,Or6,Or7,Or8,Or9,Or10,Or11,Or12]" are
        present in a molecule. Molecules with more matches are more penalized.

        Parameters
        ----------
        mol: Mol
            Mol object to evaluate.

        Returns
        -------
        int
            number of matches.
        """
        try:
            caloric_smarts = MolFromSmarts("[Or5,Or6,Or7,Or8,Or9,Or10,Or11,Or12]")
            return len(mol.GetSubstructMatches(caloric_smarts))
        except:
            return self.worst_fitness

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
                scores.append(self._predict_sweet_prob(mol)*(1 / (self._match_score(mol)+1)))
            return scores
        else:
            return [self._predict_sweet_prob(candidates)*(1 / (self._match_score(candidates)+1))]

    def method_str(self):
        """
        Get name of the evaluation function.

        Returns
        -------
        str:
            name of the evaluation function
        """
        return "Penalized Sweetness (DeepSweet-Caloric)"


class Caloric(ChemicalEvaluationFunction):
    """
    Caloric evaluation function.
    Penalizes molecules with specific groups that are related with being caloric or not.
    """

    def __init__(self, maximize: bool = True, worst_fitness: float = 0.0):
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

    def _match_score(self, mol: Mol):
        """
        Internal method to identify how many groups that match the SMARTS "[Or5,Or6,Or7,Or8,Or9,Or10,Or11,Or12]" are
        present in a molecule. Molecules with more matches are more penalized.

        Parameters
        ----------
        mol: Mol
            Mol object to evaluate.

        Returns
        -------
        List[float]
            caloric score
        """
        try:
            caloric_smarts = MolFromSmarts("[Or5,Or6,Or7,Or8,Or9,Or10,Or11,Or12]")
            n_matches = len(mol.GetSubstructMatches(caloric_smarts))
            if n_matches > 0:
                return [self.worst_fitness]
            else:
                return [1.0]
        except:
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

    def __init__(self, maximize=True, worst_fitness=0.0):
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
            return [1 - (MolLogP(mol)/25)]  # 25 is the highest logp obtained in MOSES and our generated molecules
        except:
            return [self.worst_fitness]

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

    def __init__(self, maximize=True, worst_fitness=0.0):
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
            return [qed(mol)]
        except:
            return [self.worst_fitness]

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


class MolecularWeight(ChemicalEvaluationFunction):
    """
    Molecular Weight evaluation function.
    Computes the average molecular weight of molecules.
    """

    def __init__(self,
                 min_weight: float = 300.0,
                 max_weight: float = 900,
                 maximize: bool = True,
                 worst_fitness: float = 0.0):
        """
        Initializes the MolecularWeight evaluation function.

        Parameters
        ----------
        min_weight: float
            minimum molecular weight of the molecules.
        max_weight: float
            maximum molecular weight of the molecules.
        maximize: bool
            if the goal is to maximize (True) or minimize (False) the fitness of the evaluation function.
        worst_fitness: float
            The worst fitness possible for the evaluation function.
        """
        super(MolecularWeight, self).__init__(maximize, worst_fitness)
        self.min_weight = min_weight
        self.max_weight = max_weight

    def _mol_weight(self, mol: Mol):
        """
        Computes the average molecular weight of a molecule.

        Parameters
        ----------
        mol: Mol
            Mol object to calculate the molecular weight

        Returns
        -------
        List[float]
            list with the molelular weight of the molecule
        """
        try:
            mw = MolWt(mol)
            if mw < self.min_weight:
                return [np.cos((mw - self.min_weight+200) / 320)]
            elif mw < self.max_weight:
                return [1.0]
            else:
                return [1.0 / np.log(mw / 250.0)]
        except Exception:
            return [self.worst_fitness]

    def get_fitness(self, candidates: Union[Mol, List[Mol]]):
        """
        Returns the fitness of a set of Mol objects.
        In this case it's the molecular weight score of the molecules.

        Parameters
        ----------
        candidates: Union[Mol, List[Mol]]
            Mol objects to evaluate.

        Returns
        -------
        List[int]:
            list of molecular weight scores of the candidate Mol objects.
        """
        if isinstance(candidates, list):
            scores = []
            for mol in candidates:
                scores.extend(self._mol_weight(mol))
            return scores
        else:
            return self._mol_weight(candidates)

    def method_str(self):
        """
        Get name of the evaluation function.

        Returns
        -------
        str:
            name of the evaluation function
        """
        return "MolecularWeight"


class NumberOfLargeRings(ChemicalEvaluationFunction):
    """
    Number Of Large Rings evaluation function.
    Computes the average molecular weight of molecules.
    """

    def __init__(self, maximize: bool = True, worst_fitness: float = 0.0):
        """
        Initializes the NumberOfLargeRings evaluation function.

        Parameters
        ----------
        maximize: bool
            if the goal is to maximize (True) or minimize (False) the fitness of the evaluation function.
        worst_fitness: float
            The worst fitness possible for the evaluation function.
        """
        super(NumberOfLargeRings, self).__init__(maximize, worst_fitness)

    def _ring_size(self, mol: Mol):
        """
        Computes the rings sizes of a molecule and penalizes based on the largest ring.

        Parameters
        ----------
        mol: Mol
            Mol object to calculate the largest ring score

        Returns
        -------
        List[int]
            list with the penalized value
        """
        try:
            ringsSize = [len(ring) for ring in GetSymmSSSR(mol)]

            if len(ringsSize) > 0:
                largestRing = max(ringsSize)
                if largestRing > 6:
                    return [1 / np.log((largestRing-6.0)*100)]
                else:
                    return [1.0]
            else:
                return [1.0]
        except:
            return [self.worst_fitness]

    def get_fitness(self, candidates: Union[Mol, List[Mol]]):
        """
        Returns the fitness of a set of Mol objects.
        In this case it's the largest ring penalty of the molecules.

        Parameters
        ----------
        candidates: Union[Mol, List[Mol]]
            Mol objects to evaluate.

        Returns
        -------
        List[int]:
            list of ring penalty scores of the candidate Mol objects.
        """
        if isinstance(candidates, list):
            scores = []
            for mol in candidates:
                scores.extend(self._ring_size(mol))
            return scores
        else:
            return self._ring_size(candidates)

    def method_str(self):
        """
        Get name of the evaluation function.

        Returns
        -------
        str:
            name of the evaluation function
        """
        return "NumberOfLargeRings"


class StereoisomersCounter(ChemicalEvaluationFunction):
    """
    Number Of Stereoisomers evaluation function.
    Computes the number of stereoisomers of molecules.
    """

    def __init__(self, maximize: bool = True, worst_fitness: float = 0.0):
        """
        Initializes the StereoisomersCounter evaluation function.

        Parameters
        ----------
        maximize: bool
            if the goal is to maximize (True) or minimize (False) the fitness of the evaluation function.
        worst_fitness: float
            The worst fitness possible for the evaluation function.
        """
        super(StereoisomersCounter, self).__init__(maximize, worst_fitness)

    def _chiral_count(self, mol: Mol):
        """
        Computes the chiral count of a molecule and penalizes molecules with many stereoisomers.

        Parameters
        ----------
        mol: Mol
            Mol object to calculate the chiral count

        Returns
        -------
        List[int]
            list with the penalized score
        """
        try:
            chiralCount = EnumerateStereoisomers.GetStereoisomerCount(mol,
                                                                      options=StereoEnumerationOptions(unique=True))
            if chiralCount < 5:
                return [1.0]
            else:
                return [1.0 / np.log(chiralCount * 100.0)]

        except Exception:
            return [self.worst_fitness]

    def get_fitness(self, candidates: Union[Mol, List[Mol]]):
        """
        Returns the fitness of a set of Mol objects.
        In this case it's the stereoisomers count penalty of the molecules.

        Parameters
        ----------
        candidates: Union[Mol, List[Mol]]
            Mol objects to evaluate.

        Returns
        -------
        List[int]:
            list of stereoisomers count penalty scores of the candidate Mol objects.
        """
        if isinstance(candidates, list):
            scores = []
            for mol in candidates:
                scores.extend(self._chiral_count(mol))
            return scores
        else:
            return self._chiral_count(candidates)

    def method_str(self):
        """
        Get name of the evaluation function.

        Returns
        -------
        str:
            name of the evaluation function
        """
        return "StereoisomersCounter"


class SimilarityToInitial(ChemicalEvaluationFunction):
    """
    Similarity to Initial evaluation function.
    Compares current solutions with the initial population in terms of Tanimoto Similarity.
    """

    def __init__(self, initial_population: List[str], maximize: bool = True, worst_fitness: float = 0.0):
        """
        Initializes the SimilarityToInitial evaluation function.

        Parameters
        ----------
        initial_population: List[str]
            initial population of compound' smiles to compare current compound with.
        maximize: bool
            if the goal is to maximize (True) or minimize (False) the fitness of the evaluation function.
        worst_fitness: float
            The worst fitness possible for the evaluation function.
        """
        super(SimilarityToInitial, self).__init__(maximize, worst_fitness)
        self.fingerprints = [AllChem.GetMorganFingerprint(MolFromSmiles(cmp), 2) for cmp in initial_population]

    def _compute_distance(self, mol: Mol):
        """
        Computes the distance (1 - tanimoto similarity) between the current molecule and the initial population.

        Parameters
        ----------
        mol: Mol
            Mol object to calculate the distance

        Returns
        -------
        List[int]
            list with the distance score
        """
        try:
            fp = AllChem.GetMorganFingerprint(mol, 2)
            similarities = DataStructs.BulkTanimotoSimilarity(fp, self.fingerprints)
            return [1 - max(similarities)]
        except Exception:
            return [self.worst_fitness]

    def get_fitness(self, candidates: Union[Mol, List[Mol]]):
        """
        Returns the fitness of a set of Mol objects.
        In this case it's the distance of the molecule to the initial population.

        Parameters
        ----------
        candidates: Union[Mol, List[Mol]]
            Mol objects to evaluate.

        Returns
        -------
        List[int]:
            list of distances of the candidate Mol objects.
        """
        if isinstance(candidates, list):
            scores = []
            for mol in candidates:
                scores.extend(self._compute_distance(mol))
            return scores
        else:
            return self._compute_distance(candidates)

    def method_str(self):
        """
        Get name of the evaluation function.

        Returns
        -------
        str:
            name of the evaluation function
        """
        return "SimilarityToInitial"
