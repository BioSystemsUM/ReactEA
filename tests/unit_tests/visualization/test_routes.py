import tempfile
from pathlib import Path
from unittest import TestCase

from rdkit import Chem

from reactea.visualization import Route, load_routes, draw_route, save_routes
from reactea.visualization.routes import _changed_atoms, _rule_labels, _shorten

from tests import TEST_DIR


class TestRoutes(TestCase):

    def setUp(self):
        self.results_dir = TEST_DIR / 'data' / 'output_example' / 'GA_rr_600esc_400gen_0.6sweet_0.4_caloric'
        self.routes = load_routes(self.results_dir)
        # ketone reduction (current rule ids) followed by a hydroxylation
        self.route = Route(smiles=['CC(=O)CO', 'CC(O)CO', 'CC(O)C(O)O'],
                           rule_ids=['Rule_57963', 'Rule_77233'],
                           scores={'qed': 0.4},
                           index=7)

    def test_load_routes(self):
        self.assertEqual(len(self.routes), 600)
        csv_routes = load_routes(next(self.results_dir.glob('FINAL_TRANSFORMATIONS_*.csv')))
        self.assertEqual(self.routes, csv_routes)
        for route in self.routes:
            self.assertEqual(len(route.smiles), route.n_steps + 1)
            self.assertIn('probSweet-caloric', route.scores)
        self.assertEqual(self.routes[3].n_steps, 2)
        self.assertEqual(self.routes[3].index, 3)

        with tempfile.TemporaryDirectory() as empty_dir:
            with self.assertRaises(FileNotFoundError):
                load_routes(empty_dir)

    def test_draw_route(self):
        longest = max(self.routes, key=lambda r: r.n_steps)
        unchanged = next(r for r in self.routes if r.n_steps == 0)
        invalid = Route(smiles=['CCO', 'C1CC'], rule_ids=['Rule_0'])
        for route in (self.route, longest, unchanged, invalid):
            svg = draw_route(route).svg
            self.assertTrue(svg.lstrip().startswith('<?xml') or svg.lstrip().startswith('<svg'))
        # text is drawn as glyph outlines in the SVG, so the legends are checked directly
        self.assertEqual(draw_route(self.route)._legend(2), ['Final', 'qed: 0.4'])
        self.assertEqual(draw_route(self.route, show_scores=False)._legend(2), ['Final'])
        self.assertEqual(draw_route(self.route)._legend(1), ['Step 1'])
        self.assertEqual(draw_route(unchanged)._legend(0)[0], 'Seed (unchanged)')

        with tempfile.TemporaryDirectory() as output_dir:
            drawing = draw_route(self.route, ncols=2, show_scores=False, align=False, highlight_changes=False)
            drawing.save(Path(output_dir) / 'route.svg')
            drawing.save(Path(output_dir) / 'route.png')
            self.assertTrue((Path(output_dir) / 'route.svg').read_text().strip().endswith('</svg>'))
            self.assertTrue((Path(output_dir) / 'route.png').read_bytes().startswith(b'\x89PNG'))
            with self.assertRaises(ValueError):
                drawing.save(Path(output_dir) / 'route.pdf')

    def test_save_routes(self):
        routes = self.routes[:20] + self.routes[:20]
        expected = {route.index for route in self.routes[:20] if route.n_steps > 0}
        with tempfile.TemporaryDirectory() as output_dir:
            paths = save_routes(routes, Path(output_dir) / 'routes', fmt='svg')
            self.assertEqual({p.name for p in paths}, {f'route_{i}.svg' for i in expected})
            self.assertEqual(len(list((Path(output_dir) / 'routes').iterdir())), len(expected))

    def test_changed_atoms(self):
        ketone, alcohol = Chem.MolFromSmiles('CC(=O)CO'), Chem.MolFromSmiles('CC(O)CO')
        mcs = Chem.MolFromSmarts('[#6]-[#6]-[#6]-[#8]')
        # the reduced carbon and its oxygen (gained hydrogens)
        changed = _changed_atoms(ketone, alcohol, mcs)
        self.assertEqual([alcohol.GetAtomWithIdx(i).GetSymbol() for i in changed], ['C', 'O'])

    def test_rule_labels(self):
        above, below = _rule_labels('Rule_57963')
        self.assertEqual(above, ['EC 1.14.12.12', 'EC 1.17.99.2 (+1)'])
        self.assertEqual(below[-1], 'Rule_57963')
        self.assertEqual(_rule_labels('unknown_rule'), ([], ['unknown_rule']))

    def test_shorten(self):
        self.assertEqual(_shorten('Rule_1', 20), 'Rule_1')
        self.assertEqual(len(_shorten('RR-02-c83fd078d03cc81a-02-F', 20)), 19)