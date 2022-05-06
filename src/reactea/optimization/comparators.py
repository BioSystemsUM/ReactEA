from .solution import ChemicalSolution, dominance_test


# TODO: docstrings
class ParetoDominanceComparator:
    """"""

    @staticmethod
    def compare(solution1: ChemicalSolution, solution2: ChemicalSolution):
        """"""
        if solution1 is None:
            raise Exception("The solution1 is None")
        elif solution2 is None:
            raise Exception("The solution2 is None")
        return dominance_test(solution1, solution2)
