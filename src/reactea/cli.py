import os
import time
from datetime import datetime

import click as click
from rdkit import RDLogger

from reactea.chem import Compound
from reactea.constants import ChemConstants
from reactea.io_streams import Loaders, Writers
from reactea.optimization.jmetal.ea import ChemicalEA

DATA_FILES = os.path.dirname(__file__)


def setup_configuration_file(args):
    # create dictionary from parser.parse_args()
    config_dict = vars(args)
    config_dict['time'] = datetime.now().strftime('%m-%d_%H-%M-%S')
    config_dict['start_time'] = time.time()
    return config_dict


def initialize_case_study(configs, init_pop_smiles=None):
    name = configs['case_study']
    if name == "CompoundQuality":
        from reactea.case_studies.compound_quality import CompoundQuality
        return CompoundQuality(init_pop_smiles, configs)
    elif name == "SweetReactor":
        from reactea.case_studies.sweeteners import SweetReactor
        return SweetReactor(init_pop_smiles, configs)
    else:
        raise ValueError(f"Case study {name} not found.")


@click.command()
@click.argument("config_file",
                type=click.File('r'),
                required=True,
                )
def reactea_cli(config_file):
    """Run ReactEA command line interfae.

    Mandatory arguments:

        config_file: Path to the file containing the configurations of the experiment.

        output_path: Path to the output directory.

    """
    configs = Loaders.get_config_from_yaml(config_file.name)

    # shutdown RDKit logs
    if not configs['verbose']:
        # Mute RDKit logs
        RDLogger.DisableLog("rdApp.*")

    # Load initial population
    if not configs.get("smiles"):
        init_pop, init_pop_smiles = Loaders.initialize_population(configs)
    else:
        # TODO: test this part
        init_pop_smiles = configs['smiles']
        init_pop = [ChemConstants.STANDARDIZER().standardize(Compound(init_pop_smiles[0], 'id0'))]

    # define case study
    case_study = initialize_case_study(configs, init_pop_smiles)
    # set up objective function
    objective = case_study.objective

    # get some EA parameters
    generations = configs["generations"]
    algorithm = configs["algorithm"]
    visualize = configs["visualize"]

    # set up folders
    Writers.set_up_folders(configs['output_dir'])

    # initialize reaction rules
    reaction_rules = Loaders.initialize_rules()

    # initialize objectives
    problem = objective()

    # Initialize EA
    ea = ChemicalEA(problem=problem, initial_population=init_pop, reaction_rules=reaction_rules,
                    max_generations=generations, visualizer=visualize,
                    algorithm=algorithm, configs=configs)

    # Run EA
    final_pop = ea.run()

    # Save population
    Writers.save_final_pop(final_pop, configs, case_study.feval_names())
    # Save Transformations
    Writers.save_intermediate_transformations(final_pop, configs)

    # save configs
    configs['run_time'] = time.time() - configs['start_time']
    Writers.save_configs(configs)
    print(f"Run time: {configs['run_time']} seconds!")

    # PlotResults(configs, solution_index=0).plot_results(save_fig=True)


if __name__ == "__main__":
    pass
