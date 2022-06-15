import argparse

from reactea.utilities.io import Loaders


def setup_configuration_file(args):
    # creacte dictionary from parser.parse_args()
    config_dict = vars(args)
    return config_dict

def run(configs):
    """
    Run ReactEA.
    """
    pass

def __run_cli():
    """
    Command line interface.
    All options for running ReactEA are defined here.
    """
    parser = argparse.ArgumentParser(description="Command Line Interface for running ReactEA.")
    parser.add_argument("--config_file", type=str, help="Path to the configuration file.", default=None)
    parser.add_argument("--verbose", help="Enable RDKit logs.", type=bool, default=False)
    parser.add_argument("--exp_name",
                        help="Experiment name (used to create folder with the results).",
                        type=str,
                        default='temp/')
    parser.add_argument("--smiles', help='Initial compound SMILES string to use.", type=str, default=None)
    parser.add_argument("--init_pop_path",
                        help="Path to the initial population.",
                        type=str,
                        default="/data/compounds/ecoli_sink.tsv")
    parser.add_argument("--init_pop_size",
                        help="Initial population size (default=None, uses all).",
                        type=int,
                        default=None)
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
    parser.add_argument("--case_study", help="Case study to use in the EA.", type=str, default="SweetReactor")
    args = parser.parse_args()

    if args.config_file:
        config_file = args.config_file
        configs = Loaders.get_config_from_yaml(config_file)
    else:
        configs = setup_configuration_file(args)

    run(configs)





if __name__ == "__main__":
    __run_cli()
