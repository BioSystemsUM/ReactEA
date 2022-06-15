from .solution import ChemicalSolution, dominance_test


class ParetoDominanceComparator:
    """
    Compares two solutions using aggregate constraint violation and the Pareto dominance relation as originally
    proposed by Kalyanmoy Deb.

    References:
        Deb, K., "An Efficient Constraint Handling Method for Genetic Algorithms." Computer Methods in Applied
        Mechanics and Engineering, pp. 311--338, 1998.
    """

    @staticmethod
    def compare(solution1: ChemicalSolution, solution2: ChemicalSolution):
        """
        Compares two solutions using the Pareto dominance comparator.
        Parameters
        ----------
        solution1: ChemicalSolution
            The first solution to compare.
        solution2: ChemicalSolution
            The second solution to compare.

        Returns
        -------
        int:
            -1 if solution1 is dominated by solution2,
            0 if solution1 and solution2 are non-dominated,
            1 if solution2 is dominated by solution1.
        """
        return dominance_test(solution1, solution2)
