import os

from bunch import Bunch
from rdkit import RDLogger

from reactea.case_studies.sweeteners import SweetReactor
from reactea.optimization.jmetal.ea import EA
from reactea.utils.io import Loaders, Writers

def run(configs: Bunch, case):
    # set up objective
    objective = case.objective

    # Read configurations
    generations = configs["generations"]
    algorithm = configs["algorithm"]

    # set up folders
    if not os.path.exists(f"reactea/outputs/{configs['exp_name']}/operatorLogs/"):
        os.makedirs(f"reactea/outputs/{configs['exp_name']}/operatorLogs/")

    # initialize population
    init_pop = Loaders.initialize_population(configs)

    # initialize objectives
    problem = objective(configs, multiObjective=configs['multi_objective'])

    # Initialize EA
    ea = EA(problem, initial_population=init_pop, max_generations=generations, mp=False, visualizer=False,
            algorithm=algorithm, batched=configs["batched"], configs=configs)

    # Run EA
    final_pop, unconverted_pop = ea.run()

    # Save population
    Writers.save_final_pop(final_pop, configs, case.feval_names)
    # Save Transformations
    Writers.save_intermediate_transformations(unconverted_pop, configs)

    # save configs
    Writers.save_configs(configs)


if __name__=='__main__':
    # Mute RDKit logs
    RDLogger.DisableLog("rdApp.*")

    # Load config file
    configPath = "./configs/configs_test.json"
    configs = Loaders.get_config_from_json(configPath)

    # Define the case study
    case_study = SweetReactor()

    # Run
    run(configs, case_study)

