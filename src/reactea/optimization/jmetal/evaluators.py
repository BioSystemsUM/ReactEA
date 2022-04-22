from typing import List

from jmetal.util.evaluator import Evaluator

from reactea.optimization.problem import Problem
from reactea.optimization.solution import Solution


class ChemicalEvaluator(Evaluator):

    def evaluate(self, solution_list: List[Solution], problem: Problem):
        Evaluator.evaluate_solution(solution_list, problem)
        return solution_list
