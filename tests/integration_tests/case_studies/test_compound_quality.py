import os
from unittest import TestCase

from base_test_cases import CaseStudiesBaseTestCase
from reactea.case_studies.compound_quality import CompoundQuality
from reactea.io_streams import Loaders, Writers

from reactea.optimization.jmetal.ea import ChemicalEA
from reactea.optimization.jmetal.terminators import StoppingByEvaluationsOrMeanFitnessValue


class TestCompoundQuality(CaseStudiesBaseTestCase, TestCase):

    def run_case_study(self, mo=True):
        if mo:
            self.configs['multi_objective'] = True
            self.configs['algorithm'] = 'NSGAIII'
        else:
            self.configs['multi_objective'] = False
            self.configs['algorithm'] = 'GA'

        # set up output folder
        self.output_folder = os.path.join(self.output_folder, self.configs['algorithm'])
        self.configs['output_dir'] = self.output_folder

        # initialize population and initialize population smiles
        init_pop, init_pop_smiles = Loaders.initialize_population(self.configs)
        self.assertEqual(len(init_pop), self.configs['init_pop_size'])
        self.assertEqual(len(init_pop_smiles), self.configs['init_pop_size'])

        # case study
        case_study = CompoundQuality(init_pop_smiles, self.configs)
        self.assertEqual(case_study.name(), 'CompoundQuality')

        # set up objective
        objective = case_study.objective
        self.assertTrue(objective().is_maximization)
        self.assertEqual(objective().get_name(), "ChemicalProblem")

        # initialize reaction rules
        reaction_rules = Loaders.initialize_rules()

        # set up folders
        Writers.set_up_folders(self.output_folder)

        # initialize objectives
        problem = objective()

        # Initialize EA
        ea = ChemicalEA(self.configs['algorithm'], problem, initial_population=init_pop, reaction_rules=reaction_rules,
                        max_generations=self.configs['generations'], visualizer=False, configs=self.configs)

        ea.termination_criterion = StoppingByEvaluationsOrMeanFitnessValue(0.95,
                                                                           self.configs['generations']*len(init_pop))

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
