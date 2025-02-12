import time
from datetime import datetime
from pathlib import Path

import yaml

import pandas as pd

from reactea.chem import Compound, ReactionRule

from reactea.constants import ChemConstants

DATA_FILES = Path(__file__).resolve().parent.parent / 'data'
DEEPSWEET_MOLDES = Path(__file__).parent.parent


class Loaders:
    """
    Class containing a set of input utilities
    """

    @staticmethod
    def get_config_from_yaml(yaml_file: Path):
        """
        Reads the configuration file.

        Parameters
        ----------
        yaml_file: Path
            path to yaml file

        Returns
        -------
        dict:
            dictionary containing the configurations of the experiment
        """
        with open(yaml_file, 'r') as config_file:
            config_dict = yaml.safe_load(config_file)
        config_dict['time'] = datetime.now().strftime('%m-%d_%H-%M-%S')
        config_dict['start_time'] = time.time()

        output_path = Path(config_dict['output_path']).resolve()
        config_dict['output_dir'] = output_path / config_dict['exp_name']

        init_pop_path = Path(config_dict['init_pop_path']).resolve()
        config_dict['init_pop_path'] = init_pop_path
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
        cmp_df = pd.read_csv(configs['init_pop_path'], header=0, sep='\t')
        if "compound_id" not in cmp_df.columns:
            cmp_df["compound_id"] = cmp_df.index
        if "init_pop_size" in configs:
            cmp_df = cmp_df.sample(configs["init_pop_size"])
        if "standardize" in configs and configs["standardize"]:
            return [ChemConstants.STANDARDIZER().standardize(
                Compound(row['smiles'], row["compound_id"])) for _, row in cmp_df.iterrows()], cmp_df.smiles.values
        else:
            return [Compound(row['smiles'], row["compound_id"]) for _, row in cmp_df.iterrows()], cmp_df.smiles.values

    @staticmethod
    def initialize_rules():
        """
        Loads the reaction rules.

        Returns
        -------
        List[ReactionRule]:
            list of reaction rules to use
        """

        path = DATA_FILES / 'reactionrules' / 'reaction_rules_reactea.tsv.bz2'
        rules_df = pd.read_csv(path,
                               header=0,
                               sep='\t',
                               compression='bz2')
        return [ReactionRule(row['SMARTS'], row["InternalID"], row['Reactants']) for _, row in rules_df.iterrows()]

    @staticmethod
    def load_deepsweet_ensemble():
        """
        Loads the deepsweet models tu use in the ensemble.

        Returns
        -------
        ensemble:
            deepsweet ensemble to classify compound sweetness
        """
        try:
            from deepsweet_models import DeepSweetRF, DeepSweetDNN, DeepSweetGCN, DeepSweetSVM, DeepSweetBiLSTM
            from ensemble import Ensemble
        except ImportError:
            raise ImportError("DeepSweet is not installed. Please install it to use this feature "
                              "(https://github.com/BioSystemsUM/DeepSweet).")
        models_folder_path = DEEPSWEET_MOLDES / 'evaluation_models' / 'deepsweet_models'
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