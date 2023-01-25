from typing import Union, List

from reactea.case_studies import CaseStudy
from reactea.optimization.evaluation import ChemicalEvaluationFunction, AggregatedSum
from reactea.optimization.problem import ChemicalProblem


class CaseStudyWrapper(CaseStudy):
    """
    Wrapper for Case Study classes to be used in the optimization process in ReactEA.
    """

    def __init__(self,
                 evaluation_functions: Union[ChemicalEvaluationFunction, List[ChemicalEvaluationFunction]],
                 multi_objective: bool,
                 name: str,
                 weights: List[float] = None):
        super().__init__(multi_objective)
        self.evaluation_functions = evaluation_functions
        self.multi_objective = multi_objective
        self.name = name
        self.weights = weights
        self.feval_names_str = self.feval_names()

    def objective(self):
        """
        Defines the evaluation functions to use in the optimization problem taking into account if we are facing a
        single or multi-objective problem.

        Returns
        -------
        Problem
            Problem object defining the evaluation functions of the optimization problem.
        """
        if self.multi_objective:
            problem = ChemicalProblem(self.evaluation_functions)
            return problem
        else:
            if not isinstance(self.evaluation_functions, list):
                f_ag = self.evaluation_functions
            else:
                assert len(self.evaluation_functions) == len(self.weights), \
                    "Number of weights must be equal to number of evaluation functions"
                assert sum(self.weights) == 1, "Sum of weights must be equal to 1"
                f_ag = AggregatedSum(self.evaluation_functions, self.weights)
            problem = ChemicalProblem([f_ag])
            return problem

    def name(self):
        """
        Defines the name of this Case Study.
        Returns
        -------
        str
            Name of the Case Study.
        """
        return self.name

    def feval_names(self):
        """
        Returns the names of the evaluation functions used in this Case Study.
        """
        if isinstance(self.evaluation_functions, list):
            return ';'.join([f.method_str() for f in self.evaluation_functions])
        else:
            return self.evaluation_functions.method_str()


def case_study_wrapper(evaluation_functions: Union[ChemicalEvaluationFunction, List[ChemicalEvaluationFunction]],
                       multi_objective: bool,
                       name: str,
                       weights: List[float] = None):
    """
    Wrapper for CaseStudy class to be used in the optimization process of ReactEA.

    Parameters
    ----------
    evaluation_functions: Union[ChemicalEvaluationFunction, List[ChemicalEvaluationFunction]]
        List of evaluation functions to be used in the optimization process.
    multi_objective: bool
        Boolean indicating if the optimization process is multi-objective or not.
    name: str
        Name of the case study.
    weights: List[float]
        List of weights to be used in the AggregatedSum evaluation function.

    Returns
    -------
    CaseStudyWrapper
        The wrapped CaseStudy object.
    """
    return CaseStudyWrapper(evaluation_functions, multi_objective, name, weights)
