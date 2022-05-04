from typing import List

from jmetal.util.evaluator import Evaluator, DaskEvaluator

from reactea.optimization.problem import Problem
from reactea.optimization.solution import ChemicalSolution


class ChemicalEvaluator(Evaluator):
    """
    Class representing a ChemicalEvaluator evaluator.
    Evaluates ChemicalSolutions.
    """

    def evaluate(self, solution_list: List[ChemicalSolution], problem: Problem):
        """
        Evaluates a list of Chemical Solutions using the problem evaluation functions.

        Parameters
        ----------
        solution_list: List[ChemicalSolution]
            list of chemical solutions to evaluate
        problem: Problem
            problem to evaluate the solutions

        Returns
        -------
        List[ChemicalSolutions]:
            evaluated chemical solutions
        """
        DaskEvaluator.evaluate_solution(solution_list, problem)
        return solution_list
