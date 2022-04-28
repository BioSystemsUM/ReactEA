from typing import List, Union

from reactea.optimization.problem import ChemicalProblem
from jmetal.core.problem import Problem

from reactea.optimization.solution import ChemicalSolution


class JmetalProblem(Problem[ChemicalSolution]):
    """
    Class representing a jmetal problem.
    """

    def __init__(self, problem: ChemicalProblem):
        """
        Initializes a jmetal problem.

        Parameters
        ----------
        problem: ChemicalProblem
            ChemicalProblem to use
        """
        super(JmetalProblem, self).__init__()
        self.problem = problem
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
        """
        Creates a random solution to the problem.

        Returns
        -------
        ChemicalSolution:
            random solution
        """
        raise NotImplementedError

    def _evaluate_batch(self, solutions: List[ChemicalSolution]):
        """
        Evaluates a batch of solutions.

        Parameters
        ----------
        solutions: List[ChemicalSolution]
            batch of solutions to evaluate

        Returns
        -------
        solutions: List[ChemicalSolution]
            list of evaluated solutions
        """
        list_sols = [solut.variables for solut in solutions]
        list_scores = self.problem.evaluate_solutions(list_sols)
        for i, solution in enumerate(solutions):
            for j in range(len(list_scores[i])):
                # JMetalPy only deals with minimization problems
                if self.obj_directions[j] == self.MAXIMIZE:
                    solution.objectives[j] = -1 * list_scores[i][j]
                else:
                    solution.objectives[j] = list_scores[i][j]
        return solutions

    def _evaluate_single(self, solution: ChemicalSolution):
        """
        Evaluates a single of solutions.

        Parameters
        ----------
        solution: ChemicalSolution
            solution to evaluate

        Returns
        -------
        solutions: ChemicalSolution
            evaluated solution
        """
        candidate = solution.variables
        p = self.problem.evaluate_solutions(candidate)
        for i in range(len(p)):
            # JMetalPy only deals with minimization problems
            if self.obj_directions[i] == self.MAXIMIZE:
                solution.objectives[i] = -1 * p[i]
            else:
                solution.objectives[i] = p[i]
        return solution

    def evaluate(self, solutions: Union[ChemicalSolution, List[ChemicalSolution]]):
        """
        Evaluates solutions.

        Parameters
        ----------
        solutions: Union[ChemicalSolution, List[ChemicalSolution]]
            solution or list of solutions
        Returns
        -------
        Union[ChemicalSolution, List[ChemicalSolution]]
            evaluated solution or list of solutions
        """
        return self._evaluate_batch(solutions) if isinstance(solutions, list) else self._evaluate_single(solutions)

    def get_name(self) -> str:
        """
        Get the name of the problem class.

        Returns
        -------
        str:
            name of the Chemical Problem.
        """
        return self.problem.get_name()
