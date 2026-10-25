from types import SimpleNamespace
from unittest import TestCase

import matplotlib
from matplotlib import pyplot as plt

from reactea.chem import Compound
from reactea.optimization.jmetal.observers import VisualizerObserver
from reactea.optimization.solution import ChemicalSolution


class TestVisualizerObserver(TestCase):

    def setUp(self):
        matplotlib.use('Agg')

    def tearDown(self):
        plt.close('all')

    def test_update(self):
        # jmetalpy's StreamingPlot needs matplotlib.rcsetup.interactive_bk (removed in matplotlib 3.11)
        self.assertTrue(matplotlib.rcsetup.interactive_bk)

        solutions = [ChemicalSolution(Compound(smiles, f'id{i}'), objectives)
                     for i, (smiles, objectives) in enumerate([('CCO', [-0.4, 1.0]),
                                                                ('CCCO', [-0.6, 2.0]),
                                                                ('c1ccccc1O', [-0.5, 0.0])])]
        problem = SimpleNamespace(obj_labels=['qed', 'n_rings'])
        observer = VisualizerObserver()
        for evaluations in (3, 6):
            observer.update(EVALUATIONS=evaluations, SOLUTIONS=solutions, PROBLEM=problem)
        self.assertIsNotNone(observer.figure)