from rdkit import RDLogger

from reactea.case_studies.sweeteners import SweetReactor
from reactea.optimization.jmetal.ea import ChemicalEA
from reactea.utilities.io import Loaders, Writers


# TODO: change everything to work with the constants except if they are in the configs
def run(configurations: dict, case):
    # set up objective
    objective = case.objective

    # Read configurations
    generations = configurations["generations"]
    algorithm = configurations["algorithm"]

    # set up folders
    Writers.set_up_folders(f"outputs/{configurations['exp_name']}/")

    # initialize population
    init_pop = Loaders.initialize_population(configurations)

    # initialize reaction rules
    reaction_rules, coreactants = Loaders.initialize_rules(configurations)

    # initialize objectives
    problem = objective()

    # Initialize EA
    ea = ChemicalEA(problem, initial_population=init_pop, reaction_rules=reaction_rules,
                    coreactants=coreactants, max_generations=generations, mp=False, visualizer=False,
                    algorithm=algorithm, configs=configurations)

    # Run EA
    final_pop = ea.run()

    # Save population
    Writers.save_final_pop(final_pop, configurations, case.feval_names())
    # Save Transformations
    Writers.save_intermediate_transformations(final_pop, configurations)

    # save configs
    Writers.save_configs(configurations)


if __name__ == '__main__':
    # Mute RDKit logs
    RDLogger.DisableLog("rdApp.*")

    # Load config file
    configPath = "/configs/example_config.yaml"
    configs = Loaders.get_config_from_yaml(configPath)

    # Load initial population
    init_pop_smiles = Loaders.load_initial_population_smiles(configs)

    # Define the case study
    case_study = SweetReactor(init_pop_smiles, configs)

    # Run
    run(configs, case_study)
