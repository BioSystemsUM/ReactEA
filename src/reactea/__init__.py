import warnings
from pathlib import Path
from typing import Union

from rdkit import RDLogger

from reactea.case_studies import CaseStudy
from reactea.io_streams import Loaders, Writers
from reactea.optimization.jmetal.ea import ChemicalEA
from reactea.wrappers import case_study_wrapper, evaluation_functions_wrapper

__version__ = '1.0.0'


def run_reactea(configs_path: Union[str, dict],
                case_study: CaseStudy,
                ignore_rdkit_logs: bool = True,
                ignore_warnings: bool = True):
    if ignore_rdkit_logs:
        RDLogger.DisableLog("rdApp.*")

    if ignore_warnings:
        warnings.filterwarnings("ignore")

    configs_path = Path(configs_path)
    if configs_path.exists():
        configs = Loaders.get_config_from_yaml(configs_path)
    elif isinstance(configs_path, dict):
        configs = configs_path
    else:
        raise FileNotFoundError(f"Config file {configs_path} not found.")

    # set up output folder
    output_folder = Path(configs['output_dir']) / configs['algorithm']
    configs['output_dir'] = output_folder

    # initialize population and initialize population smiles
    init_pop, init_pop_smiles = Loaders.initialize_population(configs)

    # set up objective
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
