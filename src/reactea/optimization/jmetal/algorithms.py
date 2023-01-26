from copy import deepcopy
from typing import List

import cytoolz
from jmetal.algorithm.multiobjective.nsgaiii import NSGAIII
from jmetal.algorithm.singleobjective import GeneticAlgorithm, EvolutionStrategy
from jmetal.util.constraint_handling import overall_constraint_violation_degree

from reactea.optimization.solution import ChemicalSolution


class ReactorGeneticAlgorithm(GeneticAlgorithm):
    """
    Class representing a Reactor Genetic Algorithm.
    """

    def __init__(self, **kwarg):
        """
        Initializes a ReactorGeneticAlgorithm object.
        Parameters
        ----------
        kwarg
            kwargs to use (see GeneticAlgorithm arguments)
        """
        super(ReactorGeneticAlgorithm, self).__init__(**kwarg)

    def replacement(self, population: List[ChemicalSolution], offspring_population: List[ChemicalSolution]):
        """
        Performs replacement of the less fit solutions by better solutions without repetitions (if possible)

        Parameters
        ----------
        population: List[ChemicalSolution]
            previous list of solutions
        offspring_population: List[ChemicalSolution]
            new list of solutions

        Returns
        -------
        List[ChemicalSolution]:
            new set solutions without repetitions (if possible)
        """
        population.extend(offspring_population)

        population.sort(key=lambda s: s.objectives[0])

        unique_population = list(cytoolz.unique(population, key=lambda x: x.variables.smiles))

        if len(unique_population) >= self.population_size:
            unique_population = unique_population[:self.population_size]
        else:
            unique_population.extend(population[:len(offspring_population) - len(unique_population)])

        return unique_population

    def get_result(self):
        """
        Get the EA results.

        Returns
        -------
        List[Solutions]:
            list of the EA solutions.
        """
        return self.solutions


class ReactorEvolutionStrategy(EvolutionStrategy):

    def __init__(self, **kwarg):
        """
        Initializes a ReactorEvolutionStrategy object.
        Parameters
        ----------
        kwarg
            kwargs to use (see EvolutionStrategy arguments)
        """
        super(ReactorEvolutionStrategy, self).__init__(**kwarg)

    def evaluate(self, solution_list: List[ChemicalSolution]):
        return self.population_evaluator.evaluate(deepcopy(solution_list), self.problem)

    def replacement(self,
                    population: List[ChemicalSolution],
                    offspring_population: List[ChemicalSolution]) -> List[ChemicalSolution]:
        population_pool = []

        if self.elitist:
            population_pool = population
            population_pool.extend(offspring_population)
        else:
            population_pool.extend(offspring_population)

        population_pool.sort(key=lambda s: (overall_constraint_violation_degree(s), s.objectives[0]))

        unique_population = list(cytoolz.unique(population, key=lambda x: x.variables.smiles))

        # avoid duplicates when possible
        if len(unique_population) >= self.mu:
            unique_population = unique_population[:self.mu]
        else:
            unique_population.extend(population_pool[:self.mu - len(unique_population)])

        return unique_population

    def get_result(self):
        """
        Get the EA results.

        Returns
        -------
        List[Solutions]:
            list of the EA solutions.
        """
        return self.solutions


class ReactorNSGAIII(NSGAIII):

    def __init__(self, **kwarg):
        """
        Initializes a ReactorNSGAIII object.
        Parameters
        ----------
        kwarg
            kwargs to use (see NSGAIII arguments)
        """
        super(ReactorNSGAIII, self).__init__(**kwarg)

    def get_result(self):
        """
        Get the EA results.

        Returns
        -------
        List[Solutions]:
            list of the EA solutions.
        """
        return self.solutions
