#from deepsweet_models import DeepSweetRF, DeepSweetDNN, DeepSweetGCN, DeepSweetBiLSTM, DeepSweetSVM
#from ensemble import Ensemble
import json
import os
import pickle
import random
from datetime import datetime

import pandas as pd
from bunch import Bunch
from keras.models import load_model

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../")


class Loaders:
    """"""

    @staticmethod
    def from_root(file_path):
        """"""
        return os.path.join(ROOT_DIR, file_path)

    @staticmethod
    def get_config_from_json(json_file):
        """"""
        with open(Loaders.from_root(json_file), 'r') as config_file:
            config_dict = json.load(config_file)
        configs = Bunch(config_dict)

        configs['time'] = datetime.now().strftime('%m-%d_%H-%M-%S')
        return configs

    @staticmethod
    def initialize_population(configs):
        """"""
        smiles = pd.read_csv(configs["data"]["init_pop_path"], header=None)
        smiles = list(smiles[0].values)
        random.shuffle(smiles)
        smiles = smiles[:configs["data"]["init_pop_size"]]
        return smiles

    @staticmethod
    def loadSweetModels(config):
        """"""
        # SVM
        SVM = pickle.load(open(config["svmSweet"], 'rb'))
        # RF
        RF = pickle.load(open(config["rfSweet"], 'rb'))
        # DNN
        DNN = load_model(config["dnnSweet"])
        DNN.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
        return SVM, RF, DNN

    @staticmethod
    def load_deepsweet_ensemble():
        """"""
        #models_folder_path = 'reactea/data/deep_sweet_models/'

        #list_of_models = []
        #list_of_models.append(DeepSweetRF(models_folder_path, "2d", "SelectFromModelFS"))
        #list_of_models.append(DeepSweetDNN(models_folder_path, "rdk", "all"))

        # it is necessary to insert the gpu number because it is a torch model and the device needs to be specified
        #list_of_models.append(DeepSweetGCN(models_folder_path, "cuda"))
        #list_of_models.append(DeepSweetSVM(models_folder_path, "ecfp4", "all"))
        #list_of_models.append(DeepSweetDNN(models_folder_path, "atompair_fp", "SelectFromModelFS"))
        #list_of_models.append(DeepSweetBiLSTM(models_folder_path))

        #ensemble = Ensemble(list_of_models, models_folder_path)

        #return ensemble
        return NotImplementedError


class Writers:
    """"""

    @staticmethod
    def save_final_pop(final_pop, configs, feval_names):
        # save all solutions
        destFile = f"reactea/outputs/{configs['exp_name']}/FINAL_{configs['time']}.csv"
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
    def save_intermediate_transformations(pop, configs):
        destFile = "reactea/outputs/" + configs["exp_name"] + "/FINAL_TRANSFORMATIONS_{:s}.csv".format(configs["time"])
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
    def save_configs(configs):
        configs = configs.toJSON()
        configs = json.loads(configs)
        with open(f"reactea/outputs/{configs['exp_name']}/configs.json", 'w') as outfile:
            json.dump(configs, outfile, indent=0)
