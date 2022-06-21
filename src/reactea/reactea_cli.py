import argparse
import time
from datetime import datetime

from rdkit import RDLogger

from reactea.chem.compounds import Compound
from reactea.optimization.jmetal.ea import ChemicalEA
from reactea.utilities.constants import ChemConstants
from reactea.utilities.io import Loaders, Writers
from reactea.vizualization.plot_results import PlotResults


def setup_configuration_file(args):
    # create dictionary from parser.parse_args()
    config_dict = vars(args)
    config_dict['time'] = datetime.now().strftime('%m-%d_%H-%M-%S')
    config_dict['start_time'] = time.time()
    return config_dict


def str_to_case_study(name):
    if name == "CompoundQuality":
        from reactea.case_studies.compound_quality import CompoundQuality
        return CompoundQuality
    elif name == "SweetReactor":
        from reactea.case_studies.sweeteners import SweetReactor
        return SweetReactor
    else:
        raise ValueError(f"Case study {name} not found.")


def run(configs):
    """
    Run ReactEA.
    """

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
    case_study = str_to_case_study(configs["case_study"])(init_pop_smiles, configs)
    # set up objective function
    objective = case_study.objective

    # get some EA parameters
    generations = configs["generations"]
    algorithm = configs["algorithm"]
    visualize = configs["visualize"]

    # set up folders
    Writers.set_up_folders(f"outputs/{configs['exp_name']}/")

    # initialize reaction rules
    reaction_rules, coreactants = Loaders.initialize_rules(configs)

    # initialize objectives
    problem = objective()

    # Initialize EA
    ea = ChemicalEA(problem=problem, initial_population=init_pop, reaction_rules=reaction_rules,
                    coreactants=coreactants, max_generations=generations, visualizer=visualize,
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

    PlotResults(configs, solution_index=0).plot_results(save_fig=True)


def __run_cli():
    """
    Command line interface.
    All options for running ReactEA are defined here.
    """
    parser = argparse.ArgumentParser(description="Command Line Interface for running ReactEA.")
    parser.add_argument("--config_file",
                        type=str,
                        help="Path to the configuration file. If supplied no other argument needs to be specified!",
                        default=None)
    parser.add_argument("--verbose", help="Enable RDKit logs.", type=bool, default=False)
    parser.add_argument("--exp_name",
                        help="Experiment name (used to create folder with the results).",
                        type=str,
                        default='temp/')
    parser.add_argument("--smiles", help="Initial compound SMILES string to use.", type=str, default=None)
    parser.add_argument("--init_pop_path",
                        help="Path to the initial population.",
                        type=str,
                        default="/data/compounds/ecoli_sink.tsv")
    parser.add_argument("--init_pop_size",
                        help="Initial population size (default=None, uses all).",
                        type=int,
                        default=100)
    parser.add_argument("--rules_path",
                        help="Path to the reaction rules.",
                        type=str,
                        default="/data/reactionrules/retrorules/retrorules_rr02_flat_all_forward.tsv")
    parser.add_argument("--max_rules_by_iter",
                        help="Maximum number of rules to use per iteration.",
                        type=int,
                        default=1000)
    parser.add_argument("--use_coreactant_info", help="Use coreactant information.", type=bool, default=False)
    parser.add_argument("--coreactants_path",
                        help="Path to the coreactants.",
                        type=str,
                        default="/data/reactionrules/metacycrules/metacyc_coreactants.tsv")
    parser.add_argument("--multi_objective",
                        help="Use multi-objective optimization or combine all evaluation functions to a single "
                             "objective.",
                        type=bool,
                        default=True)
    parser.add_argument("--batched", help="Use batched optimization.", type=bool, default=True)
    parser.add_argument("--generations", help="Number of generations.", type=int, default=100)
    parser.add_argument("--algorithm", help="Algorithm to use in the EA.", type=str, default="NSGAIII")
    parser.add_argument("--case_study", help="Case study to use in the EA.", type=str, default="CompoundQuality")
    parser.add_argument("--visualize", help="Visualize results.", type=bool, default=False)
    args = parser.parse_args()

    # set up configuration file
    if args.config_file:
        config_file = args.config_file
        configs = Loaders.get_config_from_yaml(config_file)
    else:
        configs = setup_configuration_file(args)

    run(configs)


if __name__ == "__main__":
    __run_cli()
