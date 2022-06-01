from unittest import TestCase

from _utils import initialize_population, initialize_rules, load_initial_population_smiles, SOURCE_DIR
from base_test_cases import CaseStudiesBaseTestCase
from reactea.case_studies.sweeteners import SweetReactor
from reactea.optimization.jmetal.ea import ChemicalEA
from reactea.utilities.io import Writers


class TestSweetReactor(CaseStudiesBaseTestCase, TestCase):

    def test_case_study(self):
        # initialize population
        init_pop = initialize_population(self.configs)

        # initialize population smiles
        init_pop_smiles = load_initial_population_smiles(self.configs)

        # case study
        case_study = SweetReactor(init_pop_smiles, self.configs)

        # set up objective
        objective = case_study.objective

        # initialize reaction rules
        reaction_rules, coreactants = initialize_rules(self.configs)

        # set up folders
        Writers.set_up_folders(self.output_folder)

        # initialize objectives
        problem = objective()

        # Initialize EA
        ea = ChemicalEA(problem, initial_population=init_pop, reaction_rules=reaction_rules,
                        coreactants=coreactants, max_generations=self.configs['generations'], mp=False,
                        visualizer=False, algorithm=self.configs['algorithm'], configs=self.configs)

        # Run EA
        final_pop = ea.run()

        # Save population
        Writers.save_final_pop(final_pop, self.configs, case_study.feval_names())
        # Save Transformations
        Writers.save_intermediate_transformations(final_pop, self.configs)

        # save configs
        Writers.save_configs(self.configs)
