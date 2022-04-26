import os
from datetime import datetime
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
    """"""

    @staticmethod
    def from_root(file_path: str):
        """"""
        return f"{ROOT_DIR}{file_path}"

    @staticmethod
    def get_config_from_json(json_file: str):
        """"""
        with open(Loaders.from_root(json_file), 'r') as config_file:
            config_dict = yaml.safe_load(config_file)
        config_dict['time'] = datetime.now().strftime('%m-%d_%H-%M-%S')
        return config_dict

    @staticmethod
    def initialize_population(configs: dict):
        """"""
        cmp_df = pd.read_csv(Loaders.from_root(configs["compounds"]["init_pop_path"]), header=0, sep='\t')
        cmp_df = cmp_df.sample(configs["compounds"]["init_pop_size"])
        return [ChemConstants.STANDARDIZER().standardize(
            Compound(row['smiles'], row["compound_id"])) for _, row in cmp_df.iterrows()]

    @staticmethod
    def initialize_rules(configs: dict):
        """"""
        rules_df = pd.read_csv(Loaders.from_root(configs["rules"]["rules_path"]), header=0, sep='\t')
        if configs["rules"]["use_coreactant_info"]:
            coreactants = Loaders.initialize_coreactants(configs)
            return [ReactionRule(row['smarts'],
                                 row["rule_id"], row["coreactants_ids"]) for _, row in rules_df.iterrows()], coreactants
        else:
            return [ReactionRule(row['smarts'], row["rule_id"]) for _, row in rules_df.iterrows()], None

    @staticmethod
    def initialize_coreactants(configs: dict):
        coreactants_df = pd.read_csv(Loaders.from_root(configs["rules"]["coreactants_path"]), header=0, set='\t')
        return [ChemConstants.STANDARDIZER().standardize(
            Compound(row['smiles'], row["compound_id"])) for _, row in coreactants_df.iterrows()]

    @staticmethod
    def load_deepsweet_ensemble():
        """"""
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


class Writers:
    """"""

    @staticmethod
    def set_up_folders(path):
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def save_final_pop(final_pop, configs: dict, feval_names):
        # save all solutions
        destFile = Loaders.from_root(f"/outputs/{configs['exp_name']}/FINAL_{configs['time']}.csv")
        configs["final_population_path"] = destFile
        with open(destFile, 'w') as f:
            f.write("SMILES;" + feval_names + "\n")
            for i, solution in enumerate(final_pop):
                f.write(str(solution.values) + ";" + ";".join([str(round(x, 3)) for x in solution.fitness]) + "\n")

        # save unique solutions
        df = pd.read_csv(destFile, sep=';', header=0)
        df = df.drop_duplicates()
        df.to_csv(destFile[:-4] + '_UNIQUE_SOLUTIONS.csv', index=False)
        configs["final_population_unique_solutions_path"] = destFile[:-4] + '_UNIQUE_SOLUTIONS.csv'

    @staticmethod
    def save_intermediate_transformations(pop, configs: dict):
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
                f.write(f"{sol.variables};{ocs};{rules}\n")

    @staticmethod
    def save_configs(configs: dict):
        with open(Loaders.from_root(f"/outputs/{configs['exp_name']}/configs.json"), 'w') as outfile:
            yaml.dump(configs, outfile)

    @staticmethod
    def update_operators_logs(configs: dict, solution: ChemicalSolution, mutant: str, rule_id: str):
        file = f"/outputs/{configs['exp_name']}/ReactionMutationLogs.txt"
        file = Loaders.from_root(file)
        objectives = []
        # TODO: check if abs makes sense for all objectives
        for obj in solution.objectives:
            objectives.append(str(abs(round(obj, 3))))
        with open(file, 'a+') as log:
            log.write(f"{solution.variables.smiles},{mutant},{rule_id},{','.join(objectives)}\n")
