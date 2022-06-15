from abc import ABC, abstractmethod
from typing import List, Union

from reactea.chem.compounds import Compound
from reactea.optimization.evaluation import ChemicalEvaluationFunction

Num = Union[int, float]


class Problem(ABC):
    """
    Base class for  representing a problem.
    A problem its characterized by its name and by a list of Evaluation Functions.
    Child classes must implement the evaluate_solutions method.
    """

    def __init__(self, name: str, fevaluation: List[ChemicalEvaluationFunction]):
        """
        Initializes a problem.

        Parameters
        ----------
        name: str
            Name of the problem class.
        fevaluation: List[ChemicalEvaluationFunction]
            list of chemical evaluation functions.
        """
        self.name = name
        if fevaluation is None:
            raise ValueError("At least one evaluation function needs to be provided")
        else:
            self.fevaluation = fevaluation
        self.number_of_objectives = len(self.fevaluation)

    @property
    def is_maximization(self):
        """
        Checks if all evaluation functions are maximization problems.

        Returns
        -------
        bool:
            True if all evaluation functions are maximization functions. False otherwise.
        """
        return all([f.maximize for f in self.fevaluation])

    @abstractmethod
    def evaluate_solutions(self, candidates: Union[Compound, List[Compound]]):
        """
        Evaluates a Compound or list of Compounds using the chemical evaluation functions.

        Parameters
        ----------
        candidates: Union[Compound, List[Compound]]
            List of Compounds to evaluate.

        Returns
        -------
        Union[List[Num], List[List[Num]]:
            fitness of the solutions for each evaluation function
        """
        raise NotImplementedError

    def __str__(self):
        return '{0} ({1} objectives)'.format(self.__class__.__name__, self.number_of_objectives)

    def __repr__(self):
        return self.__class__.__name__


class ChemicalProblem(Problem):
    """
    Class representing a Chemical Problem.
    A Chemical Problem evaluates solutions represented as Compounds.
    """

    def __init__(self, fevaluation: List[ChemicalEvaluationFunction]):
        """
        Initializes a Chemical Problem.

        Parameters
        ----------
        fevaluation: List[ChemicalEvaluationFunction]
            list of chemical evaluation functions
        """
        super(ChemicalProblem, self).__init__("ChemicalProblem", fevaluation)

    def evaluate_solutions(self, candidates: Union[Compound, List[Compound]]):
        """
        Evaluates Chemical Solutions using Chemical Evaluation Functions.

        Parameters
        ----------
        candidates: Union[Compound, List[Compound]]
            solutions to evaluate

        Returns
        -------
        Union[List[Num], List[List[Num]]:
            fitness of the solutions for each evaluation function.
        """
        if isinstance(candidates, list):
            list_mols = [smi.mol for smi in candidates]
            evals = []
            for f in self.fevaluation:
                evals.append(f(list_mols))
            return list(zip(*evals))
        else:
            candidates = candidates.mol
            evals = []
            for f in self.fevaluation:
                evals.append(f(candidates))
            return [x for xs in evals for x in xs]

    @staticmethod
    def get_name():
        """
        Get the name of the problem class.

        Returns
        -------
        str:
            name of the Chemical Problem.
        """
        return "ChemicalProblem"
