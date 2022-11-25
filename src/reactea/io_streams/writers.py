import os
from typing import List

import yaml

import pandas as pd

from reactea.io_streams import Loaders
from reactea.optimization.solution import ChemicalSolution

ROOT_DIR = os.path.dirname(__file__)[:-10]


class Writers:
    """
    Class containing a set of output utilities
    """

    @staticmethod
    def set_up_folders(path: str):
        """
        Creates folder to output results.

        Parameters
        ----------
        path: str
            path to folder to create
        """
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def save_final_pop(final_pop: List[ChemicalSolution], configs: dict, feval_names: str):
        """
        Saves final population with respective fitness in a csv file.

        Parameters
        ----------
        final_pop: List[ChemicalSolution]
            list of final solutions
        configs: dict
            configurations of the experiment
        feval_names: str
            names of the evaluation functions
        """
        # save all solutions
        destFile = Loaders.from_root(f"/outputs/{configs['exp_name']}/FINAL_{configs['time']}.csv")
        configs["final_population_path"] = destFile
        with open(destFile, 'w') as f:
            f.write("SMILES;" + feval_names + "\n")
            for i, solution in enumerate(final_pop):
                f.write(str(solution.variables.smiles) + ";" +
                        ";".join([str(round(x, 3)*-1) for x in solution.objectives]) + "\n")

        # save unique solutions
        df = pd.read_csv(destFile, sep=';', header=0)
        df = df.drop_duplicates()
        df.to_csv(destFile[:-4] + '_UNIQUE_SOLUTIONS.csv', index=False)
        configs["final_population_unique_solutions_path"] = destFile[:-4] + '_UNIQUE_SOLUTIONS.csv'

    @staticmethod
    def save_intermediate_transformations(pop: List[ChemicalSolution], configs: dict):
        """
        Saves transformations from initial compound to final compound registering the intermediate compounds
        and respective reaction rules ids.

        Parameters
        ----------
        pop: List[ChemicalSolution]
            population to save transformations
        configs: dict
            configurations of the experiment
        """
        destFile = Loaders.from_root("/outputs/" + configs["exp_name"]
                                     + "/FINAL_TRANSFORMATIONS_{:s}.csv".format(configs["time"]))
        configs["transformations_path"] = destFile
        with open(destFile, 'w') as f:
            f.write(f"FINAL_SMILES;INTERMEDIATE_SMILES;RULE_IDS\n")

            for sol in pop:
                if 'original_compound' in sol.attributes.keys():
                    ocs = sol.attributes['original_compound']
                    rules = sol.attributes['rule_id']
                else:
                    ocs = []
                    rules = []
                f.write(f"{sol.variables.smiles};{ocs};{rules}\n")

    @staticmethod
    def save_configs(configs: dict):
        """
        Saves configurations as a yaml file.
        Parameters
        ----------
        configs: dict
            configurations of the experiment
        """
        with open(Loaders.from_root(f"/outputs/{configs['exp_name']}/configs.yaml"), 'w') as outfile:
            yaml.dump(configs, outfile)

    @staticmethod
    def update_operators_logs(configs: dict, solution: ChemicalSolution, mutant: str, rule_id: str):
        """
        Updates operators logs.
        Each time a successful operator is executed, the transformation is registered.

        Parameters
        ----------
        configs: dict
            configurations of the experiment
        solution: ChemicalSolution
            solution being modified
        mutant: str
            new solution smiles
        rule_id: str
            reaction rule id
        """
        file = f"/outputs/{configs['exp_name']}/ReactionMutationLogs.txt"
        file = Loaders.from_root(file)
        objectives = []
        for obj in solution.objectives:
            objectives.append(str(round(obj, 3)*-1))
        with open(file, 'a+') as log:
            log.write(f"{solution.variables.smiles},{mutant},{rule_id},{','.join(objectives)}\n")