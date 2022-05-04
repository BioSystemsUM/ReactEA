from reactea.case_studies.case_study import CaseStudy
from reactea.optimization.evaluation import Caloric, AggregatedSum, SweetnessPredictionDeepSweet
from reactea.optimization.problem import ChemicalProblem


class SweetReactor(CaseStudy):
    """
    Sweeteners Case Study. Optimizes sweetener probability and probability of not being caloric.
    """

    def __init__(self, multi_objective: bool = False):
        """
        Initializes the sweet and non-caloric case study.

        Parameters
        ----------
        multi_objective: bool
            boolean defining if we are facing a single or multi-objective optimization problem.
        """
        super(SweetReactor, self).__init__(multi_objective)
        self.multi_objective = multi_objective

    def objective(self):
        """
        Defines the evaluation functions to use in the optimization problem (sweetness and non-caloric) taking into
        account if we are facing a single or multi-objective problem.

        Returns
        -------
        Problem
            ChemicalProblem object defining the evaluation functions, SweetnessPredictionDeepSweet and Caloric,
            of this optimization problem.
        """
        f1 = SweetnessPredictionDeepSweet()
        f2 = Caloric()
        if self.multi_objective:
            problem = ChemicalProblem([f1, f2])
            return problem
        else:
            f3 = AggregatedSum([f1, f2], [0.7, 0.3])
            problem = ChemicalProblem([f3])
            return problem

    def name(self):
        """
        Defines the name of this Case Study.

        Returns
        -------
        str
            Name of this case study.
        """
        return f"SweetenersReactor"

    def feval_names(self):
        """
        Defines the names of the evaluation functions used in this Case Study.

        Returns
        -------
        str
            Name of the evaluation functions used in this case study.
        """
        return f"probSweet;caloric" if self.multi_objective else f"probSweet-caloric"
