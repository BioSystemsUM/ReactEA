from typing import List

import cytoolz
from jmetal.algorithm.singleobjective import GeneticAlgorithm

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

    def reproduction(self, mating_population: List[ChemicalSolution]) -> List[ChemicalSolution]:
        number_of_parents_to_combine = self.crossover_operator.get_number_of_parents()

        offspring_population = []
        for i in range(0, self.offspring_population_size, number_of_parents_to_combine):
            parents = []
            for j in range(number_of_parents_to_combine):
                parents.append(mating_population[i + j])

            offspring = self.crossover_operator.execute(parents)

            for solution in offspring:
                self.mutation_operator.execute(solution)
                offspring_population.append(solution)
                if len(offspring_population) >= self.offspring_population_size:
                    break

        return offspring_population

    def get_result(self):
        """
        Get the EA results.

        Returns
        -------
        List[Solutions]:
            list of the EA solutions.
        """
        return self.solutions
