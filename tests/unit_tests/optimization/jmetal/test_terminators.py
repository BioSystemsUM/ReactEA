from types import SimpleNamespace
from unittest import TestCase

from jmetal.core.problem import Problem

from reactea.chem import Compound
from reactea.optimization.jmetal.terminators import StoppingByEvaluations, StoppingByEvaluationsOrImprovement, \
    StoppingByEvaluationsOrMeanFitnessValue, StoppingByScore, ReactEATerminationCriterion
from reactea.optimization.solution import ChemicalSolution

# a maximized (e.g. qed) and a minimized (e.g. n_rings) objective
PROBLEM = SimpleNamespace(obj_directions=[Problem.MAXIMIZE, Problem.MINIMIZE])


def solutions(*scores):
    # jmetalpy objectives (maximized ones negated) and the raw scores
    population = []
    for i, (qed, n_rings) in enumerate(scores):
        solution = ChemicalSolution(Compound('CCO', i), [-qed, n_rings])
        solution.attributes['scores'] = [qed, n_rings]
        population.append(solution)
    return population


def update(criterion, *scores):
    criterion.update(SOLUTIONS=solutions(*scores), PROBLEM=PROBLEM)
    return criterion.is_met


class TestTerminators(TestCase):

    def test_stopping_by_evaluations(self):
        criterion = StoppingByEvaluations(3)
        self.assertEqual([update(criterion, (0.5, 1)) for _ in range(3)], [False, False, True])

    def test_stopping_by_improvement(self):
        criterion = StoppingByEvaluationsOrImprovement(patience=1, max_generations=100)
        self.assertFalse(update(criterion, (0.5, 2)))
        # improves in the minimized objective only
        self.assertFalse(update(criterion, (0.4, 1)))
        self.assertFalse(update(criterion, (0.4, 1)))
        self.assertTrue(update(criterion, (0.5, 1)))

    def test_stopping_by_mean_fitness_value(self):
        criterion = StoppingByEvaluationsOrMeanFitnessValue(0.5, max_generations=100)
        # the mean of the minimized objective (0.6) is above the value
        self.assertFalse(update(criterion, (0.6, 1.0), (0.6, 0.2)))
        self.assertTrue(update(criterion, (0.6, 0.4), (0.6, 0.2)))

    def test_stopping_by_score(self):
        maximized = StoppingByScore(0.9, objective=0, max_generations=100)
        self.assertFalse(update(maximized, (0.5, 3), (0.8, 3)))
        self.assertTrue(update(maximized, (0.5, 3), (0.95, 3)))

        minimized = StoppingByScore(0, objective=1, max_generations=100)
        self.assertFalse(update(minimized, (0.5, 3), (0.8, 1)))
        self.assertTrue(update(minimized, (0.5, 0), (0.8, 1)))

    def test_single_solution_and_scores_from_objectives(self):
        # SA and LS pass a single solution; without the 'scores' attribute, they come from the objectives
        criterion = StoppingByScore(0.9, max_generations=100)
        criterion.update(SOLUTIONS=ChemicalSolution(Compound('CCO', 0), [-0.95, 1]), PROBLEM=PROBLEM)
        self.assertTrue(criterion.is_met)

    def test_custom_criterion(self):
        class StopWhenAllRingFree(ReactEATerminationCriterion):

            def stop(self, scores, maximize):
                return all(n_rings == 0 for _, n_rings in scores)

        criterion = StopWhenAllRingFree(max_generations=100)
        self.assertFalse(update(criterion, (0.5, 0), (0.5, 1)))
        self.assertTrue(update(criterion, (0.5, 0), (0.5, 0)))
        self.assertEqual(criterion.generations, 2)
