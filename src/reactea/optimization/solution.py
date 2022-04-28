import copy
from abc import ABC, abstractmethod
from typing import Union, List

from jmetal.core.solution import Solution

from reactea.chem.compounds import Compound

Num = Union[int, float]


class ChemicalSolutionInterface(ABC):
    """
    Base Chemical Solution Interface.
    Chemical Solutions must implement the get_fitness and get_representation methods.
    """

    @abstractmethod
    def get_fitness(self):
        """
        Gets the fitness of the solution.

        Returns
        -------
        List[Num]
            List of fitness of each objective.
        """
        raise NotImplementedError

    @abstractmethod
    def get_representation(self):
        """
        Gets the representation of the solution.

        Returns
        -------
        str:
            Representation of the solution.
        """
        raise NotImplementedError


class ChemicalSolution(Solution, ChemicalSolutionInterface):
    """
    Class representing chemical solutions.
    A Chemical Solution is represented by a Compound object and respective objectives' fitness.
    """

    def __init__(self, variables: Compound, objectives: List[Num] = None, is_maximization: bool = True):
        """
        Initializes a Chemical Solution.

        Parameters
        ----------
        variables: Compound
            Compound Object
        objectives : List[Num]
            Solution objectives' fitness.
        is_maximization: bool
            If it is a maximization or minimization problem.
        """
        super(ChemicalSolution, self).__init__(1, len(objectives))
        if objectives is None:
            objectives = [0.0]
        self.variables = variables
        self.objectives = objectives
        self.attributes = {}
        self._is_maximize = is_maximization

    def get_fitness(self):
        """
        Get list of the solution objectives' fitness.

        Returns
        -------
        List[Num]:
            List of the solution objectives' fitness.
        """
        return self.objectives

    def get_representation(self):
        """
        Get the representation of the solution.

        Returns
        -------
        str:
            SMILES string of the solution.
        """
        return self.variables.smiles

    def __str__(self):
        return f"Solution ({self.variables.smiles}, {self.objectives}, {self.attributes})"

    def __eq__(self, solution):
        return self.variables.smiles == solution.variables.smiles

    def __gt__(self, solution):
        if isinstance(solution, self.__class__):
            return dominance_test(self, solution, maximize=self._is_maximize) == 1
        return False

    def __lt__(self, solution):
        if isinstance(solution, self.__class__):
            return dominance_test(self, solution, maximize=self._is_maximize) == -1
        return False

    def __ge__(self, solution):
        if isinstance(solution, self.__class__):
            return dominance_test(self, solution, maximize=self._is_maximize) != -1
        return False

    def __le__(self, solution):
        if isinstance(solution, self.__class__):
            return dominance_test(self, solution, maximize=self._is_maximize) != 1
        return False

    def __copy__(self):
        values = copy.copy(self.variables)
        fitness = self.objectives
        new_solution = ChemicalSolution(values, fitness)
        return new_solution

    def __hash__(self):
        return hash(str(self.variables))


# TODO: best place to put this functions
def dominance_test(solution1: ChemicalSolution, solution2: ChemicalSolution, maximize: bool = True):
    """
    Tests Pareto dominance.

    Parameters
    ----------
    solution1: ChemicalSolution
        The first solution to compare.
    solution2: ChemicalSolution
        The second solution to compare.
    maximize: bool
        If it is maximization (True) or minimization (False).
    Returns
    -------
    int:
       1 : if the first solution dominates the second;
       -1 : if the second solution dominates the first;
       0 : if none of the solutions dominates the other.
    """
    best_is_one = 0
    best_is_two = 0

    values1 = solution1.get_fitness()
    values2 = solution2.get_fitness()

    for i in range(len(values1)):
        value1 = values1[i]
        value2 = values2[i]
        if value1 != value2:
            if value1 < value2:
                best_is_two = 1
            if value1 > value2:
                best_is_one = 1

    if best_is_one > best_is_two:
        if maximize:
            result = 1
        else:
            result = -1
    elif best_is_two > best_is_one:
        if maximize:
            result = -1
        else:
            result = 1
    else:
        result = 0

    return result


def non_dominated_population(population: List[ChemicalSolution], maximize: bool = True, filter_duplicate: bool = True):
    """
    Returns the non dominated solutions from the population.

    Parameters
    ----------
    population: List[ChemicalSolution]
        List of Chemical Solutions.
    maximize: bool
        Maximization (True) or Minimization (False) problem.
    filter_duplicate: bool
        Filter (True) or keep (False) duplicates.

    Returns
    -------
    List[ChemicalSolution]:
        List of non dominated solutions from the population.
    """
    """
    Returns the non dominated solutions from the population.
    """
    non_dominated = []
    for i in range(len(population) - 1):
        individual = population[i]
        j = 0
        dominates = True
        while j < len(population) and dominates:
            if dominance_test(individual, population[j], maximize=maximize) == -1:
                dominates = False
            else:
                j += 1
        if dominates:
            non_dominated.append(individual)

    if filter_duplicate:
        result = filter_duplicates(non_dominated)
    else:
        result = non_dominated
    return result


def filter_duplicates(population):
    """
    Filters duplicated solutions from the population.

    Parameters
    ----------
    population: List[ChemicalSolutions]
        list of chemical solutions to filter

    Returns
    -------
    List[ChemicalSolutions]:
        list of chemical solutions without duplicates.
    """

    def remove_equal(ind: ChemicalSolution, pop: List[ChemicalSolution]):
        """
        Removes a chemical solution from list of chemical solutions.

        Parameters
        ----------
        ind: ChemicalSolution
            Chemical Solution to remove.
        pop: List[ChemicalSolution]
            list of chemical solutions to filter.

        Returns
        -------
        List[ChemicalSolution]:
            filtered list of chemical solutions
        """
        filtered = []
        for other in pop:
            if ind != other:
                filtered.append(other)
        return filtered

    filtered_list = []
    l = population
    while len(l) > 1:
        individual = l[0]
        filtered_list.append(individual)
        l = remove_equal(individual, l)
    if l:
        filtered_list.extend(l)
    return filtered_list


def cmetric(pf1: List[ChemicalSolution], pf2: List[ChemicalSolution], maximize: bool = True):
    """
    Computes the c-metric quality indicator.

    Parameters
    ----------
    pf1: List[ChemicalSolution]
        The first pareto front.
    pf2: List[ChemicalSolution]
        The second pareto front.
    maximize: bool
        If it is a maximization (True) or minimization (False) problem.
    Returns
    -------
    float, float, float, float
        r1: percentage of solutions on pf2 dominated by some solution on pf1;
        r2: percentage of solutions on pf1 dominated by some solution on pf2;
        pf1_2: solutions on pf2 dominated by some solution on pf1;
        pf2_1: solutions on pf1 dominated by some solution on pf2.
    """
    # solutions on pf2 dominated by some solution on pf1
    pf1_2 = set()
    # solutions on pf1 dominated by some solution on pf2
    pf2_1 = set()
    for s1 in pf1:
        for s2 in pf2:
            d = dominance_test(s1, s2, maximize=maximize)
            if d == 1:
                pf1_2.add(s2)
            elif d == -1:
                pf2_1.add(s1)
    r1 = len(pf1_2) / len(pf2)
    r2 = len(pf2_1) / len(pf1)
    return r1, r2, pf1_2, pf2_1
