from reactea.case_studies.case_study import CaseStudy
from reactea.optimization.evaluation import Caloric, AggregatedSum, SweetnessPredictionDeepSweet
from reactea.optimization.problem import ChemicalProblem


class SweetReactor(CaseStudy):
    """"""

    def objective(self, configs: dict, multi_objective: bool = False):
        """"""
        f1 = SweetnessPredictionDeepSweet(configs)
        f2 = Caloric()
        if multi_objective:
            problem = ChemicalProblem([f1, f2], configs)
            return problem
        else:
            f3 = AggregatedSum([f1, f2], [0.7, 0.3])
            problem = ChemicalProblem([f3], configs)
            return problem

    def name(self):
        """"""
        return f"SweetenersReactor"

    def feval_names(self):
        """"""
        return f"probSweet;caloric"
