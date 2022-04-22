from typing import List

from reactea.optimization.problem import ChemicalProblem
from jmetal.core.problem import Problem

from reactea.optimization.solution import Solution


class JmetalProblem(Problem[Solution]):
    """"""

    def __init__(self, problem: ChemicalProblem, batched: bool = True):
        """"""
        super(JmetalProblem, self).__init__()
        self.problem = problem
        self.batched = batched
        self.number_of_objectives = len(self.problem.fevaluation)
        self.obj_directions = []
        self.obj_labels = []
        for f in self.problem.fevaluation:
            self.obj_labels.append(str(f))
            if f.maximize:
                self.obj_directions.append(self.MAXIMIZE)
            else:
                self.obj_directions.append(self.MINIMIZE)

    def create_solution(self):
        """"""
        raise NotImplementedError

    def _evaluate_batch(self, solutions: List[Solution]):
        """"""
        list_sols = [solut.variables for solut in solutions]
        list_scores = self.problem.evaluate_solution(list_sols, self.batched)
        for i, solution in enumerate(solutions):
            for j in range(len(list_scores[i])):
                # JMetalPy only deals with minimization problems
                if self.obj_directions[j] == self.MAXIMIZE:
                    solution.objectives[j] = -1 * list_scores[i][j]
                else:
                    solution.objectives[j] = list_scores[i][j]
        return solutions

    def _evaluate_single(self, solution: Solution):
        """"""
        candidate = solution.variables
        p = self.problem.evaluate_solution(candidate, self.batched)
        for i in range(len(p)):
            # JMetalPy only deals with minimization problems
            if self.obj_directions[i] == self.MAXIMIZE:
                solution.objectives[i] = -1 * p[i]
            else:
                solution.objectives[i] = p[i]
        return solution

    def evaluate(self, solutions):
        return self._evaluate_batch(solutions) if self.batched else self._evaluate_single(solutions)

    def get_name(self) -> str:
        return self.problem.get_name()
