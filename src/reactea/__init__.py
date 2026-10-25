import warnings
from pathlib import Path
from typing import Union

from rdkit import RDLogger

from reactea.case_studies import CaseStudy
from reactea.io_streams import Loaders, Writers
from reactea.optimization.jmetal.ea import ChemicalEA
from reactea.visualization import load_routes, save_routes
from reactea.wrappers import case_study_wrapper, evaluation_functions_wrapper

__version__ = '2.0.0'


def run_reactea(configs_path: Union[str, dict],
                case_study: CaseStudy,
                ignore_rdkit_logs: bool = True,
                ignore_warnings: bool = True):
    """
    Runs ReactEA: optimizes the seed compounds for the case study and saves the results.

    The results are saved in `<output_path>/<exp_name>/<algorithm>/`.

    Parameters
    ----------
    configs_path: Union[str, dict]
        path to the configuration file (yaml) or dictionary with the configurations
    case_study: CaseStudy
        case study (evaluation functions) to optimize
    ignore_rdkit_logs: bool
        mute the RDKit logs (True) or not (False)
    ignore_warnings: bool
        ignore the Python warnings (True) or not (False)
    """
    if ignore_rdkit_logs:
        RDLogger.DisableLog("rdApp.*")

    if ignore_warnings:
        warnings.filterwarnings("ignore")

    if isinstance(configs_path, dict):
        configs = Loaders.process_configs(dict(configs_path))
    elif Path(configs_path).exists():
        configs = Loaders.get_config_from_yaml(Path(configs_path))
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
                    visualizer=configs.get('visualize', False),
                    configs=configs)

    # Run EA
    final_pop = ea.run()

    # Save population
    Writers.save_final_pop(final_pop, configs, case_study.feval_names())
    # Save Transformations
    Writers.save_intermediate_transformations(final_pop, configs)

    # save configs
    Writers.save_configs(configs)

    # draw the transformation routes
    if configs.get('save_routes', False):
        save_routes(load_routes(configs['transformations_path']), Path(configs['output_dir']) / 'routes')
