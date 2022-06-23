from unittest import TestCase

from _utils import initialize_population, initialize_rules, load_initial_population_smiles
from base_test_cases import CaseStudiesBaseTestCase
from reactea.case_studies.sweeteners import SweetReactor
from reactea.optimization.jmetal.ea import ChemicalEA
from reactea.utilities.io import Writers


class TestSweetReactor(CaseStudiesBaseTestCase, TestCase):

    def run_case_study(self, mo=True):
        if mo:
            self.configs['multi_objective'] = True
            self.configs['algorithm'] = 'NSGAIII'
        else:
            self.configs['multi_objective'] = False
            self.configs['algorithm'] = 'GA'

        # initialize population
        init_pop = initialize_population(self.configs)
        self.assertEqual(len(init_pop), self.configs['init_pop_size'])

        # initialize population smiles
        init_pop_smiles = load_initial_population_smiles(self.configs)
        self.assertEqual(len(init_pop_smiles), self.configs['init_pop_size'])

        # case study
        case_study = SweetReactor(init_pop_smiles, self.configs)
        self.assertEqual(case_study.name(), 'SweetenersReactor')

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
                        coreactants=coreactants, max_generations=self.configs['generations'],
                        visualizer=False, algorithm=self.configs['algorithm'], configs=self.configs)

        # Run EA
        final_pop = ea.run()
        self.assertIsInstance(final_pop, list)

        # Save population
        Writers.save_final_pop(final_pop, self.configs, case_study.feval_names())
        # Save Transformations
        Writers.save_intermediate_transformations(final_pop, self.configs)

        # save configs
        Writers.save_configs(self.configs)

    def test_case_study(self):
        self.run_case_study()
        self.run_case_study(mo=False)

