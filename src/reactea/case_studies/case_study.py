from abc import ABC, abstractmethod


class CaseStudy(ABC):
    """
    Base class for all Case Studies.
    A Case Study defines the evaluation functions to use in the optimization problem.
    """

    def __init__(self, configs: dict, multi_objective: bool = False):
        """
        Initializes the case study at a class level.

        Parameters
        ----------
         configs : dict
            dictionary with the experiment configurations.
        multi_objective: bool
            boolean defining if we are facing a single or multi-objective optimization problem.
        """
        self.configs = configs
        self.multi_objective = multi_objective

    @abstractmethod
    def objective(self):
        """
        Defines the evaluation functions to use in the optimization problem taking into account if we are facing a
        single or multi-objective problem.

        Returns
        -------
        Problem
            Problem object defining the evaluation functions of the optimization problem.
        """
        raise NotImplementedError

    @abstractmethod
    def name(self):
        """
        Defines the name of the Case Study.

        Returns
        -------
        str
            Name of the case study.
        """
        return NotImplementedError

    @abstractmethod
    def feval_names(self):
        """
        Defines the names of the evaluation functions used in the Case Study.

        Returns
        -------
        str
            Name of the evaluation functions used in the case study.
        """
        return NotImplementedError
