from unittest import TestCase
from unittest.mock import patch

from rdkit.Chem import MolFromSmiles

from reactea.case_studies.sweeteners import SweetReactor
from reactea.optimization.evaluation import PenalizedSweetness, SimilarityToInitial


class FakeEnsemble:

    def predict(self, mols):
        return [[0.8] * len(mols)]


@patch('reactea.optimization.evaluation.Loaders.load_deepsweet_ensemble', return_value=FakeEnsemble())
class TestDeepSweetLoading(TestCase):
    """
    Runs without DeepSweet (its models are replaced by a fake ensemble).
    """

    def test_models_are_loaded_once(self, load_ensemble):
        f = PenalizedSweetness()
        mols = [MolFromSmiles('CCO'), MolFromSmiles('OCC(O)CO')]
        for _ in range(3):
            scores = f.get_fitness(mols)

        self.assertEqual(load_ensemble.call_count, 1)
        self.assertEqual(len(scores), len(mols))

    def test_sweet_reactor_single_objective(self, load_ensemble):
        case_study = SweetReactor(['CCO'], {'multi_objective': False})
        aggregated = case_study.objective().fevaluation[0]

        self.assertEqual(len(aggregated.fevaluation), len(aggregated.tradeoffs))
        self.assertEqual(aggregated.tradeoffs, [0.5, 0.15, 0.1, 0.05, 0.05, 0.15])
        self.assertIsInstance(aggregated.fevaluation[-1], SimilarityToInitial)
