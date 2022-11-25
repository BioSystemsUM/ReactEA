import os
import time
from datetime import datetime
from typing import List

import yaml

import pandas as pd

from deepsweet_models import DeepSweetRF, DeepSweetDNN, DeepSweetGCN, DeepSweetSVM, DeepSweetBiLSTM
from ensemble import Ensemble
from reactea.chem.compounds import Compound
from reactea.chem.reaction_rules import ReactionRule
from reactea.optimization.solution import ChemicalSolution
from .constants import ChemConstants

ROOT_DIR = os.path.dirname(__file__)[:-10]


class Loaders:
    """
    Class containing a set of input utilities
    """

    @staticmethod
    def from_root(file_path: str):
        """
        Gets path of file from root.

        Parameters
        ----------
        file_path: str
            file path

        Returns
        -------
        str:
            file path from root
        """
        return f"{ROOT_DIR}/{file_path}"

    @staticmethod
    def get_config_from_yaml(yaml_file: str):
        """
        Reads the configuration file.

        Parameters
        ----------
        yaml_file: str
            path to yaml file

        Returns
        -------
        dict:
            dictionary containing the configurations of the experiment
        """
        with open(Loaders.from_root(yaml_file), 'r') as config_file:
            config_dict = yaml.safe_load(config_file)
        config_dict['time'] = datetime.now().strftime('%m-%d_%H-%M-%S')
        config_dict['start_time'] = time.time()
        return config_dict

    @staticmethod
    def initialize_population(configs: dict):
        """
        Loads the initial population.

        Parameters
        ----------
        configs: dict
            configurations of the experiment (containing path to initial population file)

        Returns
        -------
        List[Compound]:
            list of compounds to use as initial population
        """
        cmp_df = pd.read_csv(Loaders.from_root(configs["init_pop_path"]), header=0, sep='\t')
        cmp_df = cmp_df.sample(configs["init_pop_size"])
        return [ChemConstants.STANDARDIZER().standardize(
            Compound(row['smiles'], row["compound_id"])) for _, row in cmp_df.iterrows()], cmp_df.smiles.values

    @staticmethod
    def initialize_rules():
        """
        Loads the reaction rules.

        Parameters
        ----------
        configs: dict
            configurations of the experiment (containing path to reaction rules file)

        Returns
        -------
        List[ReactionRule]:
            list of reaction rules to use
        """
        rules_df = pd.read_csv(Loaders.from_root('/data/reactionrules/reaction_rules_reactea.tsv.bz2'),
                               header=0,
                               sep='\t',
                               compression='bz2')
        return [ReactionRule(row['SMARTS'], row["InternalID"], row['Reactants']) for _, row in rules_df.iterrows()]

    @staticmethod
    def initialize_coreactants(configs: dict, standardize: bool = False):
        """
        Loads the set of coreactants

        Parameters
        ----------
        configs: dict
            configurations of the experiment (containing path to coreactants file)
        standardize: bool
            whether to standardize the coreactants

        Returns
        -------
        List[Compound]:
            list of compounds to use as coreactants
        """
        coreactants_df = pd.read_csv(Loaders.from_root(configs["coreactants_path"]), header=0, sep='\t')
        if standardize:
            return [ChemConstants.STANDARDIZER().standardize(
                Compound(row['smiles'], row["compound_id"])) for _, row in coreactants_df.iterrows()]
        else:
            return [Compound(row['smiles'], row["compound_id"]) for _, row in coreactants_df.iterrows()]

    @staticmethod
    def load_deepsweet_ensemble():
        """
        Loads the deepsweet models tu use in the ensemble.

        Returns
        -------
        ensemble:
            deepsweet ensemble to classify compound sweetness
        """
        models_folder_path = Loaders.from_root('/evaluation_models/deepsweet_models/')
        list_of_models = [DeepSweetRF(models_folder_path, "2d", "SelectFromModelFS"),
                          DeepSweetDNN(models_folder_path, "rdk", "all"),
                          # it is necessary to insert the gpu number because it is a torch model and the device needs
                          # to be specified
                          DeepSweetGCN(models_folder_path, "cuda"),
                          DeepSweetSVM(models_folder_path, "ecfp4", "all"),
                          DeepSweetDNN(models_folder_path, "atompair_fp", "SelectFromModelFS"),
                          DeepSweetBiLSTM(models_folder_path)]

        ensemble = Ensemble(list_of_models, models_folder_path)
        return ensemble

    @staticmethod
    def load_results_case(index: int, configs: dict):
        """
        Loads the results file.

        Parameters
        ----------
        index: int
            index of the case to load
        configs: dict
            configurations of the experiment (containing path to results file)

        Returns
        -------
        pandas.DataFrame:
            dataframe containing the results
        """
        return pd.read_csv(configs["transformations_path"], header=0, sep=';').iloc[index]


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
