from abc import ABC, abstractmethod
from functools import reduce
from typing import List, Union

import numpy as np
from joblib import Parallel, delayed
from rdkit import DataStructs
from rdkit.Chem import MolFromSmarts, Mol, GetSymmSSSR, EnumerateStereoisomers, AllChem, MolFromSmiles, MolToSmiles
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

    def __init__(self, maximize: bool = True, worst_fitness: float = 0.0):
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

    def get_fitness(self, candidates: Union[Mol, List[Mol]]):
        """
        Evaluates the fitness of the candidate(s).

        Parameters
        ----------
        candidates: Union[Mol, List[Mol]]
            The candidate(s) to evaluate.

        Returns
        -------
        List[float]
            The fitness(es) of the candidate(s).
        """
        if isinstance(candidates, Mol):
            candidates = [candidates]
        return Parallel(n_jobs=-1, backend="multiprocessing")(delayed(self.get_fitness_single)(candidate)
                                                              for candidate in candidates)

    @abstractmethod
    def get_fitness_single(self, candidate: Mol):
        """
        Get fitness of a single solution.

        Parameters
        ----------
        candidate: Mol
            Mol object to get fitness from.
        Returns
        -------
        float
            Fitness of the Mol object
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
            Function to compute the worst fitness from the passed ChemicalEvaluationFunctions
            (choose from mean, max, min).
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
        Evaluates the fitness of the candidate(s).

        Parameters
        ----------
        candidates: Union[Mol, List[Mol]]
            The candidate(s) to evaluate.

        Returns
        -------
        List[float]
            The fitness(es) of the candidate(s).
        """
        if "Sweetness" in self.method_str():
            return self.get_fitness_single(candidates)
        if isinstance(candidates, Mol):
            candidates = [candidates]
        return Parallel(n_jobs=-1, backend="multiprocessing")(delayed(self.get_fitness_single)(candidate)
                                                              for candidate in candidates)

    def get_fitness_single(self, candidate: Mol):
        """
        Returns the fitness of a single Mol object.
        In this case it's the weighted fitness of multiple evaluation functions.

        Parameters
        ----------
        candidate: Mol
            Mol object to evaluate.

        Returns
        -------
        float:
            fitness of the Mol object.
        """
        evals = []
        for f in self.fevaluation:
            evals.append(f.get_fitness_single(candidate))
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

    def __init__(self, maximize: bool = True, worst_fitness: float = 0.0):
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

    def _predict_sweet_prob(self, candidates: List[Mol]):
        """
        Computes the predicted sweetness probability of a list of Mol objects.

        Parameters
        ----------
        candidates: List[Mol]

        Returns
        -------
        List[float]:
            predicted sweetness probability of the candidates.
        """
        return self.ensemble.predict(candidates)[0]

    def get_fitness(self, candidates: Union[Mol, List[Mol]]):
        """
        Computes the predicted sweetness probability of a list of Mol objects.
        Handles invalid molecules.

        Parameters
        ----------
        candidates: Union[Mol, List[Mol]]
            Mol object or list of Mol objects to get the fitness from.

        Returns
        -------
        List[float]:
            predicted sweetness probability of the candidates.
        """
        if isinstance(candidates, Mol):
            candidates = [candidates]
        invalid = [i for i in range(len(candidates)) if not self.invalid_candidate(candidates[i])]
        candidates = [x for i, x in enumerate(candidates) if i not in invalid]
        scores = self._predict_sweet_prob(candidates)
        for i in invalid:
            scores = np.insert(scores, i, 0.0)
        scores = [np.max([0, i]) for i in scores]
        return scores

    def get_fitness_single(self, candidate: Mol):
        """
        Returns the fitness of a single Mol object.
        Same as get_fitness (alias).
        """
        return self.get_fitness(candidate)

    @staticmethod
    def invalid_candidate(candidate: Mol):
        """
        Returns whether a candidate is invalid or not for the Deepsweet ensemble prediction.

        Parameters
        ----------
        candidate: Mol
            Mol object to check.

        Returns
        -------
        bool:
            True if the candidate is invalid, False otherwise.
        """
        if isinstance(candidate, Mol):
            if len(MolToSmiles(candidate)) <= 128:
                return True
        return False

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

    def __init__(self, maximize: bool = True, worst_fitness: float = 0.0):
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

    def get_fitness(self, candidates: Union[Mol, List[Mol]]):
        """
        Computes the predicted penalized sweetness probability of a list of Mol objects.
        Sweetness prediction is penalized by the presence of groups that confer caloric value to the molecule.

        Parameters
        ----------
        candidates: Union[Mol, List[Mol]]
            Mol object or list of Mol objects to get the fitness from.

        Returns
        -------
        List[float]:
            predicted penalized sweetness probability of the candidates.
        """
        return np.multiply(SweetnessPredictionDeepSweet().get_fitness(candidates), Caloric().get_fitness(candidates))

    def get_fitness_single(self, candidate: Mol):
        """
        Returns the fitness of a single Mol object.
        Same as get_fitness (alias).
        """
        return self.get_fitness(candidate)

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
        float
            caloric score
        """
        try:
            caloric_smarts = MolFromSmarts("[Or5,Or6,Or7,Or8,Or9,Or10,Or11,Or12]")
            n_matches = len(mol.GetSubstructMatches(caloric_smarts))
            return 1 / (n_matches + 1)
        except:
            return self.worst_fitness

    def get_fitness_single(self, candidate: Mol):
        """
        Returns the fitness of a single Mol object.
        In this case it's the caloric score of the molecule.

        Parameters
        ----------
        candidate: Mol
            Mol object to evaluate.

        Returns
        -------
        float:
            caloric score of the candidate Mol object.
        """
        return self._match_score(candidate)

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

    def __init__(self, maximize: bool = True, worst_fitness: float = 0.0):
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
        float
            partition coefficient of the molecule
        """
        try:
            return 1 - (MolLogP(mol)/25)  # 25 is the highest logp obtained in MOSES and our generated molecules
        except:
            return self.worst_fitness

    def get_fitness_single(self, candidate: Mol):
        """
        Returns the fitness of a single Mol object.
        In this case it's the partition coefficient of the molecule.

        Parameters
        ----------
        candidate: Mol
            Mol object to evaluate.

        Returns
        -------
        float:
            partition coefficients of the candidate Mol object.
        """
        return self._get_logp(candidate)

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

    def __init__(self, maximize: bool = True, worst_fitness: float = 0.0):
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
        float:
            drug-likeliness score of the molecule
        """
        try:
            return qed(mol)
        except:
            return self.worst_fitness

    def get_fitness_single(self, candidate: Mol):
        """
        Returns the fitness of a single Mol object.
        In this case it's the drug-likeliness score of the molecule.

        Parameters
        ----------
        candidate: Mol
            Mol object to evaluate.

        Returns
        -------
        float:
            drug-likeliness scores of the candidate Mol object.
        """
        return self._qed(candidate)

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
        float:
            molecular weight of the molecule
        """
        try:
            mw = MolWt(mol)
            if mw < self.min_weight:
                return np.cos((mw - self.min_weight+200) / 320)
            elif mw < self.max_weight:
                return 1.0
            else:
                return 1.0 / np.log(mw / 250.0)
        except Exception:
            return self.worst_fitness

    def get_fitness_single(self, candidate: Mol):
        """
        Returns the fitness of a single Mol object.
        In this case it's the molecular weight score of the molecule.

        Parameters
        ----------
        candidate: Mol
            Mol object to evaluate.

        Returns
        -------
        float:
            molecular weight score of the candidate Mol object.
        """
        return self._mol_weight(candidate)

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
        float:
            penalized ring size score
        """
        try:
            ringsSize = [len(ring) for ring in GetSymmSSSR(mol)]

            if len(ringsSize) > 0:
                largestRing = max(ringsSize)
                if largestRing > 6:
                    return 1 / np.log((largestRing-6.0)*100)
                else:
                    return 1.0
            else:
                return 1.0
        except:
            return self.worst_fitness

    def get_fitness_single(self, candidate: Mol):
        """
        Returns the fitness of a set of Mol object.
        In this case it's the largest ring penalty of the molecule.

        Parameters
        ----------
        candidate: Mol
            Mol object to evaluate.

        Returns
        -------
        float:
            ring penalty scores of the candidate Mol object.
        """
        return self._ring_size(candidate)

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
        int
            penalized chiral count score
        """
        try:
            chiralCount = EnumerateStereoisomers.GetStereoisomerCount(mol,
                                                                      options=StereoEnumerationOptions(unique=True))
            if chiralCount < 5:
                return 1.0
            else:
                return 1.0 / np.log(chiralCount * 100.0)

        except Exception:
            return self.worst_fitness

    def get_fitness_single(self, candidate: Mol):
        """
        Returns the fitness of a single Mol object.
        In this case it's the stereoisomers count penalty of the molecule.

        Parameters
        ----------
        candidate: Mol
            Mol object to evaluate.

        Returns
        -------
        int:
            stereoisomers count penalty score of the candidate Mol object.
        """
        return self._chiral_count(candidate)

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
        Computes the distance (1 - Tanimoto similarity) between the current molecule and the initial population.

        Parameters
        ----------
        mol: Mol
            Mol object to calculate the distance

        Returns
        -------
        int:
            distance score
        """
        try:
            fp = AllChem.GetMorganFingerprint(mol, 2)
            similarities = DataStructs.BulkTanimotoSimilarity(fp, self.fingerprints)
            return 1 - max(similarities)
        except Exception:
            return self.worst_fitness

    def get_fitness_single(self, candidate: Mol):
        """
        Returns the fitness of a single Mol object.
        In this case it's the distance of the molecule to the initial population.

        Parameters
        ----------
        candidate: Mol
            Mol object to evaluate.

        Returns
        -------
        int:
            distance of the candidate Mol object.
        """
        return self._compute_distance(candidate)

    def method_str(self):
        """
        Get name of the evaluation function.

        Returns
        -------
        str:
            name of the evaluation function
        """
        return "SimilarityToInitial"
