import tempfile
from pathlib import Path
from unittest import TestCase

import yaml

from reactea.chem import Compound
from reactea.io_streams import Writers
from reactea.optimization.solution import ChemicalSolution


class TestWriters(TestCase):

    def test_save_configs(self):
        with tempfile.TemporaryDirectory() as output_dir:
            configs = {'output_dir': Path(output_dir), 'init_pop_path': Path('data/seeds.tsv'), 'generations': 2}
            Writers.save_configs(configs)
            saved = yaml.safe_load((Path(output_dir) / 'configs.yaml').read_text())
        self.assertEqual(saved, {'output_dir': output_dir, 'init_pop_path': 'data/seeds.tsv', 'generations': 2})
        # the configs in memory are not changed
        self.assertIsInstance(configs['output_dir'], Path)

    def test_update_operators_logs(self):
        # a maximized (qed) and a minimized (n_rings) objective: jmetalpy stores the first one negated
        solution = ChemicalSolution(Compound('CCO', 'id0'), [-0.4, 2.0])
        solution.attributes['scores'] = [0.4, 2.0]
        with tempfile.TemporaryDirectory() as output_dir:
            Writers.update_operators_logs({'output_dir': Path(output_dir)}, solution, 'CCCO', 'Rule_1')
            log = (Path(output_dir) / 'ReactionMutationLogs.txt').read_text()
        self.assertEqual(log, 'CCO,CCCO,Rule_1,0.4,2.0\n')