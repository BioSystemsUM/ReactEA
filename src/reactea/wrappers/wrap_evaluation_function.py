from rdkit.Chem import Mol

from reactea.optimization.evaluation import ChemicalEvaluationFunction


class EvaluationFunctionWrapper(ChemicalEvaluationFunction):
    """
    Wrapper for the evaluation function to be used in the optimization process in ReactEA.
    """

    def __init__(self, evaluation_function, maximize, worst_fitness, name):
        super().__init__(maximize, worst_fitness)
        self.evaluation_function = evaluation_function
        self.name = name

    def get_fitness_single(self, candidate: Mol):
        """
        Returns the fitness of a single candidate.

        Parameters
        ----------
        candidate: Mol
            The candidate to evaluate.

        Returns
        -------
        float:
            The fitness of the candidate.
        """
        try:
            return self.evaluation_function(candidate)
        except :
            return self.worst_fitness

    def method_str(self):
        """
        Returns the name of the evaluation function.
        """
        return self.name


def evaluation_functions_wrapper(function: callable,
                                 maximize: bool,
                                 worst_fitness: float,
                                 name: str):
    """
    Wraps a function to be used as an evaluation function for the optimization process.

    Parameters
    ----------
    function: callable
        The function to be wrapped.
    maximize: bool
        Whether the function should be maximized or minimized.
    worst_fitness: float
        The worst fitness value that the function can return.
    name: str
        The name of the function.

    Returns
    -------
    EvaluationFunctionWrapper
        The wrapped function.
    """
    return EvaluationFunctionWrapper(function, maximize, worst_fitness, name)
