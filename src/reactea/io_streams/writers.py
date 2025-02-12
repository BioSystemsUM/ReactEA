from pathlib import Path
from typing import List

import yaml

import pandas as pd

from reactea.optimization.solution import ChemicalSolution


class Writers:
    """
    Class containing a set of output utilities
    """

    @staticmethod
    def set_up_folders(path: Path):
        """
        Creates folder to output results.

        Parameters
        ----------
        path: str
            path to folder to create
        """
        path.mkdir(parents=True, exist_ok=True)

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
        destFile = configs['output_dir'] / f"FINAL_{configs['time']}.csv"
        configs["final_population_path"] = destFile
        with destFile.open('w') as f:
            f.write("SMILES;" + feval_names + "\n")
            for i, solution in enumerate(final_pop):
                f.write(str(solution.variables.smiles) + ";" +
                        ";".join([str(round(x, 3)*-1) for x in solution.objectives]) + "\n")

        # save unique solutions
        df = pd.read_csv(destFile, sep=';', header=0)
        df = df.drop_duplicates()
        unique_solutions_path = destFile.stem + '_UNIQUE_SOLUTIONS.csv'
        df.to_csv(destFile.parent / unique_solutions_path, index=False)
        configs["final_population_unique_solutions_path"] = destFile.parent / unique_solutions_path

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
        destFile = Path(configs['output_dir']) / f"FINAL_TRANSFORMATIONS_{configs['time']}.csv"
        configs["transformations_path"] = destFile
        with destFile.open('w') as f:
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
        destFile = configs['output_dir'] / 'configs.yaml'
        with destFile.open('w') as outfile:
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
        destFile = configs['output_dir'] / 'ReactionMutationLogs.txt'
        objectives = [str(round(obj, 3) * -1) for obj in solution.objectives]
        with destFile.open('a+') as log:
            log.write(f"{solution.variables.smiles},{mutant},{rule_id},{','.join(objectives)}\n")
