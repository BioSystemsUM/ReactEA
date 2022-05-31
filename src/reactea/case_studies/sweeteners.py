from reactea.case_studies.case_study import CaseStudy
from reactea.optimization.evaluation import Caloric, AggregatedSum, SweetnessPredictionDeepSweet, PenalizedSweetness, \
    MolecularWeight, NumberOfLargeRings, StereoisomersCounter, LogP, SimilarityToInitial
from reactea.optimization.problem import ChemicalProblem
from reactea.utilities.io import Loaders


class SweetReactor(CaseStudy):
    """
    Sweeteners Case Study.
    Optimizes sweetener probability with penalization of being caloric, molecular weight, number of large rings,
    stereoisomers count, logP and similarity to initial.
    """

    def __init__(self, configs):
        """
        Initializes the sweet and non-caloric case study.

        Parameters
        ----------
        multi_objective: bool
            boolean defining if we are facing a single or multi-objective optimization problem.
        """
        super(SweetReactor, self).__init__(configs['multi_objective'])
        self.multi_objective = configs['multi_objective']
        self.population_smiles = Loaders.load_initial_population_smiles(configs)

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
        f1 = PenalizedSweetness()
        f2 = MolecularWeight()
        f3 = NumberOfLargeRings()
        f4 = StereoisomersCounter()
        f5 = LogP()
        f6 = SimilarityToInitial(self.population_smiles)
        if self.multi_objective:
            f_ag = AggregatedSum([f2, f3, f4, f5], [0.3, 0.3, 0.1, 0.3])
            problem = ChemicalProblem([f1, f_ag, f6])
            return problem
        else:
            f_ag = AggregatedSum([f1, f2, f3, f4, f5], [0.5, 0.15, 0.1, 0.05, 0.05, 0.15])
            problem = ChemicalProblem([f_ag])
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
