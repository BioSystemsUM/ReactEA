import os
from datetime import datetime

import pandas as pd
import yaml

from reactea.chem.compounds import Compound
from reactea.utilities.constants import ChemConstants

ROOT_DIR = os.path.dirname(__file__)
SOURCE_DIR = os.path.dirname(__file__)[:-6]


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
    return f"{ROOT_DIR}{file_path}"


def load_config(file_path):
    with open(file_path, 'r') as config_file:
        config_dict = yaml.safe_load(config_file)
    config_dict['time'] = datetime.now().strftime('%m-%d_%H-%M-%S')
    return config_dict


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
    cmp_df = pd.read_csv(from_root(configs["init_pop_path"]), header=0, sep='\t')
    cmp_df = cmp_df.sample(configs["init_pop_size"])
    return [ChemConstants.STANDARDIZER().standardize(
        Compound(row['smiles'], row["compound_id"])) for _, row in cmp_df.iterrows()]


def load_initial_population_smiles(configs: dict):
    """
    Loads the initial population smiles.

    Parameters
    ----------
    configs: dict
        configurations of the experiment (containing path to initial population file)

    Returns
    -------
    List[str]:
        list of compounds' smiles used as initial population
    """
    cmp_df = pd.read_csv(from_root(configs["init_pop_path"]), header=0, sep='\t')
    cmp_df = cmp_df.sample(configs["init_pop_size"])
    return cmp_df.smiles.values
