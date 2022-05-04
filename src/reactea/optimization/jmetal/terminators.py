import numpy as np
from jmetal.util.termination_criterion import TerminationCriterion


class StoppingByEvaluationsOrMeanFitnessValue(TerminationCriterion):
    """
    StoppingByEvaluationsOrFitnessValue termination criterion.
    Stops EA if maximum number of evaluations is met or if the mean of the objectives reaches a value.
    """

    def __init__(self, expected_value: float, max_evaluations: int):
        """
        Initializes a StoppingByEvaluationsOrFitnessValue stopping criterion.

        Parameters
        ----------
        expected_value: float
            maximum fitness value to reach
        max_evaluations: int
            maximum number of evaluations
        """
        super(StoppingByEvaluationsOrMeanFitnessValue, self).__init__()
        self.expected_value = expected_value
        self.max_evaluations = max_evaluations
        self.value = 0.0
        self.evaluations = 0

    def update(self, *args, **kwargs):
        """
        Updates the number current number of iterations and fitness value.
        """
        self.evaluations = kwargs["EVALUATIONS"]
        solutions = kwargs["SOLUTIONS"]
        mean_fit = np.mean([s.objecives for s in solutions])
        self.value = mean_fit

    @property
    def is_met(self):
        """
        Checks if maximum number of evaluations or fitness value are reached.

        Returns
        -------
        bool
            true if the value or number of evaluations are reached, false otherwise
        """
        return self.value >= self.expected_value or self.evaluations >= self.max_evaluations
