from rdkit import RDLogger

from reactea.case_studies.sweeteners import SweetReactor
from reactea.optimization.jmetal.ea import EA
from reactea.utilities.io import Loaders, Writers

def run(configs: dict, case):
    # set up objective
    objective = case.objective

    # Read configurations
    generations = configs["generations"]
    algorithm = configs["algorithm"]

    # set up folders
    Writers.set_up_folders(f"outputs/{configs['exp_name']}/")

    # initialize population
    init_pop = Loaders.initialize_population(configs)

    # initialize reaction rules
    reaction_rules, coreactants = Loaders.initialize_rules(configs)

    # initialize objectives
    problem = objective()

    # Initialize EA
    ea = EA(problem, initial_population=init_pop, reaction_rules=reaction_rules, coreactants=coreactants,
            max_generations=generations, mp=False, visualizer=False, algorithm=algorithm, configs=configs)

    # Run EA
    final_pop = ea.run()

    # Save population
    Writers.save_final_pop(final_pop, configs, case.feval_names())
    # Save Transformations
    Writers.save_intermediate_transformations(final_pop, configs)

    # save configs
    Writers.save_configs(configs)


if __name__ == '__main__':
    # Mute RDKit logs
    RDLogger.DisableLog("rdApp.*")

    # Load config file
    configPath = "/configs/example_config.yaml"
    configs = Loaders.get_config_from_json(configPath)

    # Define the case study
    case_study = SweetReactor(configs['multi_objective'])

    # Run
    run(configs, case_study)

