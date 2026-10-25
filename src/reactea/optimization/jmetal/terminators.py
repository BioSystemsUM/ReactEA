from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from jmetal.core.problem import Problem
from jmetal.util.termination_criterion import TerminationCriterion


class ReactEATerminationCriterion(TerminationCriterion, ABC):
    """
    Base class of the ReactEA termination criteria.
    Stops the EA after `max_generations` generations or when the `stop` method returns True.

    Child classes only need to implement the `stop` method, which receives the scores of the current solutions
    (as returned by the evaluation functions, both for maximized and minimized objectives) at each generation.
    """

    def __init__(self, max_generations: Optional[int] = None):
        """
        Initializes the termination criterion.

        Parameters
        ----------
        max_generations: Optional[int]
            maximum number of generations (if None, the number of generations of the configuration is used)
        """
        super(ReactEATerminationCriterion, self).__init__()
        self.max_generations = max_generations
        self.generations = 0
        self.stopped = False

    @staticmethod
    def solution_scores(solution, maximize: List[bool]) -> List[float]:
        """
        Gets the scores of a solution as returned by the evaluation functions.

        Parameters
        ----------
        solution: ChemicalSolution
            evaluated solution
        maximize: List[bool]
            whether each objective is maximized (True) or minimized (False)

        Returns
        -------
        List[float]:
            scores of the solution
        """
        if 'scores' in solution.attributes:
            return list(solution.attributes['scores'])
        # jmetalpy only minimizes, so the objectives of maximized functions are negated
        return [-o if m else o for o, m in zip(solution.objectives, maximize)]

    def update(self, *args, **kwargs):
        """
        Updates the termination criterion at each generation.

        Parameters
        ----------
        args:
            args to use
        kwargs:
            kwargs to use (SOLUTIONS and PROBLEM are used)
        """
        solutions = kwargs['SOLUTIONS']
        # single-solution algorithms (SA and LS) pass a single solution
        if not isinstance(solutions, list):
            solutions = [solutions]
        maximize = [direction == Problem.MAXIMIZE for direction in kwargs['PROBLEM'].obj_directions]
        scores = [self.solution_scores(solution, maximize) for solution in solutions]
        self.generations += 1
        self.stopped = bool(self.stop(scores, maximize))

    @abstractmethod
    def stop(self, scores: List[List[float]], maximize: List[bool]) -> bool:
        """
        Checks if the EA must stop (called at each generation).

        Parameters
        ----------
        scores: List[List[float]]
            scores of each solution of the current population (one value per objective)
        maximize: List[bool]
            whether each objective is maximized (True) or minimized (False)

        Returns
        -------
        bool:
            True to stop the EA, False to continue
        """
        raise NotImplementedError

    @property
    def is_met(self):
        """
        Checks if the maximum number of generations was reached or if the `stop` method returned True.

        Returns
        -------
        bool
            true if the EA must stop, false otherwise
        """
        max_reached = self.max_generations is not None and self.generations >= self.max_generations
        return self.stopped or max_reached


class StoppingByEvaluations(ReactEATerminationCriterion):
    """
    StoppingByEvaluations termination criterion.
    Stops the EA when the maximum number of generations is reached.
    """

    def stop(self, scores: List[List[float]], maximize: List[bool]) -> bool:
        return False


class StoppingByEvaluationsOrImprovement(ReactEATerminationCriterion):
    """
    StoppingByEvaluationsOrImprovement termination criterion.
    Stops the EA if the maximum number of generations is reached or if the mean score of the population does not
    improve (in any objective) for `patience` generations.
    """

    def __init__(self, patience: int, max_generations: Optional[int] = None):
        """
        Initializes a StoppingByEvaluationsOrImprovement termination criterion.

        Parameters
        ----------
        patience: int
            maximum number of generations without improvement
        max_generations: Optional[int]
            maximum number of generations
        """
        super(StoppingByEvaluationsOrImprovement, self).__init__(max_generations)
        self.patience = patience
        self.best_mean = None
        self.no_improvement = 0

    def stop(self, scores: List[List[float]], maximize: List[bool]) -> bool:
        mean = np.mean(scores, axis=0)
        if self.best_mean is None:
            self.best_mean = mean
            return False
        improved = [m > b if is_max else m < b for m, b, is_max in zip(mean, self.best_mean, maximize)]
        if any(improved):
            self.best_mean = np.where(maximize, np.maximum(mean, self.best_mean), np.minimum(mean, self.best_mean))
            self.no_improvement = 0
        else:
            self.no_improvement += 1
        return self.no_improvement > self.patience


class StoppingByEvaluationsOrMeanFitnessValue(ReactEATerminationCriterion):
    """
    StoppingByEvaluationsOrMeanFitnessValue termination criterion.
    Stops the EA if the maximum number of generations is reached or if the mean score of the population reaches a
    value in every objective (at least the value for maximized objectives, at most for minimized ones).
    """

    def __init__(self, expected_value: float, max_generations: Optional[int] = None):
        """
        Initializes a StoppingByEvaluationsOrMeanFitnessValue termination criterion.

        Parameters
        ----------
        expected_value: float
            mean score to reach
        max_generations: Optional[int]
            maximum number of generations
        """
        super(StoppingByEvaluationsOrMeanFitnessValue, self).__init__(max_generations)
        self.expected_value = expected_value

    def stop(self, scores: List[List[float]], maximize: List[bool]) -> bool:
        mean = np.mean(scores, axis=0)
        return all(m >= self.expected_value if is_max else m <= self.expected_value
                   for m, is_max in zip(mean, maximize))


class StoppingByScore(ReactEATerminationCriterion):
    """
    StoppingByScore termination criterion.
    Stops the EA if the maximum number of generations is reached or if a solution reaches a score in an objective
    (at least the value for maximized objectives, at most for minimized ones).
    """

    def __init__(self, value: float, objective: int = 0, max_generations: Optional[int] = None):
        """
        Initializes a StoppingByScore termination criterion.

        Parameters
        ----------
        value: float
            score to reach
        objective: int
            index of the objective (in the order of the case study's evaluation functions)
        max_generations: Optional[int]
            maximum number of generations
        """
        super(StoppingByScore, self).__init__(max_generations)
        self.value = value
        self.objective = objective

    def stop(self, scores: List[List[float]], maximize: List[bool]) -> bool:
        objective_scores = [s[self.objective] for s in scores]
        if maximize[self.objective]:
            return max(objective_scores) >= self.value
        return min(objective_scores) <= self.value
