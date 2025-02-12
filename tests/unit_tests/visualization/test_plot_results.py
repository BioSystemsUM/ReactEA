from unittest import TestCase

from reactea.vizualization.plot_results import PlotResults

from tests import TEST_DIR


class TestPlotResults(TestCase):

    def test_plot_results(self):
        path = TEST_DIR / 'data' / 'output_example' / 'GA_rr_600esc_400gen_0.6sweet_0.4_caloric' / 'FINAL_TRANSFORMATIONS_04-29_16-51-54.csv'
        mock_output_configs = {'transformations_path': path}
        PlotResults(mock_output_configs, solution_index=0).plot_results(save_fig=False)
        PlotResults(mock_output_configs, solution_index=4).plot_results(save_fig=False)
        PlotResults(mock_output_configs, solution_index=6).plot_results(save_fig=False)
