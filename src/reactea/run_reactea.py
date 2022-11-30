import os.path

from rdkit import RDLogger

from reactea.case_studies.sweeteners import SweetReactor
from reactea.io_streams import Writers, Loaders
from reactea.optimization.jmetal.ea import ChemicalEA


def _run(configurations: dict, case, init_pop):
    # set up objective
    objective = case.objective

    # Read configurations
    generations = configurations["generations"]
    algorithm = configurations["algorithm"]

    # set up folders
    Writers.set_up_folders(f"outputs/{configurations['exp_name']}/")

    # initialize reaction rules
    reaction_rules = Loaders.initialize_rules()

    # initialize objectives
    problem = objective()

    # Initialize EA
    ea = ChemicalEA(problem, initial_population=init_pop, reaction_rules=reaction_rules,
                    max_generations=generations, visualizer=True,
                    algorithm=algorithm, configs=configurations)

    # Run EA
    final_pop = ea.run()

    # Save population
    Writers.save_final_pop(final_pop, configurations, case.feval_names())
    # Save Transformations
    Writers.save_intermediate_transformations(final_pop, configurations)

    # save configs
    Writers.save_configs(configurations)


def run():
    # Mute RDKit logs
    RDLogger.DisableLog("rdApp.*")

    # Load config file
    configPath = f"{os.getcwd()}/configs/example_config.yaml"
    configs = Loaders.get_config_from_yaml(configPath)
    configs['output_dir'] = f"{os.getcwd()}/{configs['output_path']}{configs['exp_name']}/"

    # Load initial population
    init_pop, init_pop_smiles = Loaders.initialize_population(configs)

    # Define the case study
    case_study = SweetReactor(init_pop_smiles, configs)

    # Run
    _run(configs, case_study, init_pop)


if __name__ == '__main__':
    run()
