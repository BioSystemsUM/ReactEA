import os
import warnings
from typing import Union

from rdkit import RDLogger

from reactea.case_studies import CaseStudy
from reactea.io_streams import Loaders, Writers
from reactea.optimization.jmetal.ea import ChemicalEA
from reactea.wrappers import case_study_wrapper, evaluation_functions_wrapper

ROOT_DIR = os.path.dirname(__file__)


def run_reactea(configs_path: Union[str, dict],
                case_study: CaseStudy,
                ignore_rdkit_logs: bool = True,
                ignore_warnings: bool = True):
    if ignore_rdkit_logs:
        RDLogger.DisableLog("rdApp.*")

    if ignore_warnings:
        warnings.filterwarnings("ignore")

    if isinstance(configs_path, str) and os.path.exists(configs_path):
        configs = Loaders.get_config_from_yaml(configs_path)
    else:
        configs = configs_path

    # set up output folder
    output_folder = os.path.join(configs['output_dir'], configs['algorithm'])
    configs['output_dir'] = output_folder

    # initialize population and initialize population smiles
    init_pop, init_pop_smiles = Loaders.initialize_population(configs)

    # set up objective
    case_study = case_study()
    objective = case_study.objective

    # initialize reaction rules
    reaction_rules = Loaders.initialize_rules()

    # set up folders
    Writers.set_up_folders(output_folder)

    # initialize objectives
    problem = objective()

    # Initialize EA
    ea = ChemicalEA(configs['algorithm'],
                    problem,
                    initial_population=init_pop,
                    reaction_rules=reaction_rules,
                    max_generations=configs['generations'],
                    visualizer=False,
                    configs=configs)

    # Run EA
    final_pop = ea.run()

    # Save population
    Writers.save_final_pop(final_pop, configs, case_study.feval_names())
    # Save Transformations
    Writers.save_intermediate_transformations(final_pop, configs)

    # save configs
    Writers.save_configs(configs)
