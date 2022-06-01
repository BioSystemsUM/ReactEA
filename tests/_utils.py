import os
from datetime import datetime

import pandas as pd
import yaml

from reactea.chem.compounds import Compound
from reactea.chem.reaction_rules import ReactionRule
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
    cmp_df = pd.read_csv(from_root(configs["compounds"]["init_pop_path"]), header=0, sep='\t')
    cmp_df = cmp_df.sample(configs["compounds"]["init_pop_size"])
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
        list of compound' smiles used as initial population
    """
    cmp_df = pd.read_csv(from_root(configs["compounds"]["init_pop_path"]), header=0, sep='\t')
    cmp_df = cmp_df.sample(configs["compounds"]["init_pop_size"])
    return cmp_df.smiles.values


def initialize_rules(configs: dict):
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
    rules_df = pd.read_csv(from_root(configs["rules"]["rules_path"]), header=0, sep='\t')
    if configs["rules"]["use_coreactant_info"]:
        coreactants = initialize_coreactants(configs)
        return [ReactionRule(row['smarts'],
                             row["rule_id"], row["coreactants_ids"]) for _, row in rules_df.iterrows()], coreactants
    else:
        return [ReactionRule(row['smarts'], row["rule_id"]) for _, row in rules_df.iterrows()], None


def initialize_coreactants(configs: dict):
    """
    Loads the set of coreactants

    Parameters
    ----------
    configs: dict
        configurations of the experiment (containing path to coreactants file)

    Returns
    -------
    List[Compound]:
        list of compounds to use as coreactants
    """
    coreactants_df = pd.read_csv(from_root(configs["rules"]["coreactants_path"]), header=0, set='\t')
    return [ChemConstants.STANDARDIZER().standardize(
        Compound(row['smiles'], row["compound_id"])) for _, row in coreactants_df.iterrows()]