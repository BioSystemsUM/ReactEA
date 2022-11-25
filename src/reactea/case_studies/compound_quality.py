from typing import List

from reactea.case_studies import CaseStudy
from reactea.optimization.evaluation import MolecularWeight, LogP, QED, AggregatedSum
from reactea.optimization.problem import ChemicalProblem


class CompoundQuality(CaseStudy):
    """
    Compound Quality Case Study.
    Optimizes CompoundQuality, in specific MolecularWeight, LogP and QED.
    """

    def __init__(self, initial_population: List[str], configs: dict):
        """
        Initializes the CompoundQuality case study.

        Parameters
        ----------
        initial_population: List[str]
            List of SMILES strings used as initial population.
        configs: dict
            dictionary with the experiment configurations.
        """
        super(CompoundQuality, self).__init__(configs['multi_objective'])
        self.multi_objective = configs['multi_objective']
        self.population_smiles = initial_population
        self.feval_names_str = None

    def objective(self):
        """
        Defines the evaluation functions to use in the optimization problem (CompoundQuality) taking into
        account if we are facing a single or multi-objective problem.

        Returns
        -------
        Problem
            ChemicalProblem object defining the evaluation functions, MolecularWeight, LogP and QED,
            of this optimization problem.
        """
        f1 = MolecularWeight()
        f2 = LogP()
        f3 = QED()

        if self.multi_objective:
            problem = ChemicalProblem([f1, f2, f3])
            self.feval_names_str = f"{f1.method_str()};{f2.method_str()};{f3.method_str()}"
            return problem
        else:
            f_ag = AggregatedSum([f1, f2, f3], [0.25, 0.25, 0.5])
            problem = ChemicalProblem([f_ag])
            self.feval_names_str = f"{f_ag.method_str()}"
            return problem

    def name(self):
        """
        Defines the name of this Case Study.

        Returns
        -------
        str
            Name of this case study.
        """
        return f"CompoundQuality"

    def feval_names(self):
        """
        Defines the names of the evaluation functions used in this Case Study.

        Returns
        -------
        str
            Name of the evaluation functions used in this case study.
        """
        return self.feval_names_str
