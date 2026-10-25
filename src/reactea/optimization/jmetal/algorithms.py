from copy import deepcopy
from typing import List

import cytoolz
import numpy as np
from jmetal.algorithm.multiobjective import IBEA
from jmetal.algorithm.multiobjective.nsgaiii import NSGAIII
from jmetal.algorithm.singleobjective import GeneticAlgorithm, EvolutionStrategy
from jmetal.core.quality_indicator import EpsilonIndicator
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

    def result(self):
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

        unique_population = list(cytoolz.unique(population_pool, key=lambda x: x.variables.smiles))

        # avoid duplicates when possible
        if len(unique_population) >= self.mu:
            unique_population = unique_population[:self.mu]
        else:
            unique_population.extend(population_pool[:self.mu - len(unique_population)])

        return unique_population

    def result(self):
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

    def result(self):
        """
        Get the EA results.

        Returns
        -------
        List[Solutions]:
            list of the EA solutions.
        """
        return self.solutions


class ReactorIBEA(IBEA):

    def __init__(self, **kwarg):
        """
        Initializes a ReactorIBEA object.
        Parameters
        ----------
        kwarg
            kwargs to use (see IBEA arguments)
        """
        super(ReactorIBEA, self).__init__(**kwarg)

    def compute_fitness_values(self, population: List[ChemicalSolution], kappa: float) -> List[ChemicalSolution]:
        """
        Computes the IBEA fitness of each solution.
        Overridden because jmetalpy's EpsilonIndicator expects numpy arrays instead of lists.

        Parameters
        ----------
        population: List[ChemicalSolution]
            list of solutions
        kappa: float
            fitness scaling factor

        Returns
        -------
        List[ChemicalSolution]:
            solutions with the 'fitness' attribute updated
        """
        for i in range(len(population)):
            population[i].attributes["fitness"] = 0

            for j in range(len(population)):
                if j != i:
                    epsilon = EpsilonIndicator(np.array([population[i].objectives]))
                    population[i].attributes["fitness"] += -np.exp(
                        -epsilon.compute(np.array([population[j].objectives])) / kappa
                    )
        return population

    def replacement(self, population: List[ChemicalSolution],
                    offspring_population: List[ChemicalSolution]) -> List[ChemicalSolution]:
        """
        Environmental selection of IBEA.
        Overridden because jmetalpy's EpsilonIndicator expects numpy arrays instead of lists.

        Parameters
        ----------
        population: List[ChemicalSolution]
            current population
        offspring_population: List[ChemicalSolution]
            offspring population

        Returns
        -------
        List[ChemicalSolution]:
            the new population
        """
        join_population = population + offspring_population
        join_population_size = len(join_population)
        join_population = self.compute_fitness_values(join_population, self.kappa)

        while join_population_size > self.population_size:
            current_fitnesses = [individual.attributes["fitness"] for individual in join_population]
            index_worst = current_fitnesses.index(min(current_fitnesses))

            for i in range(join_population_size):
                epsilon = EpsilonIndicator(np.array([join_population[i].objectives]))
                join_population[i].attributes["fitness"] += np.exp(
                    -epsilon.compute(np.array([join_population[index_worst].objectives])) / self.kappa
                )

            join_population.pop(index_worst)
            join_population_size = join_population_size - 1

        return join_population
