from unittest import TestCase

from reactea.vizualization.plot_results import PlotResults


class TestPlotResults(TestCase):

    def test_plot_results(self):
        mock_output_configs = {'transformations_path': '/home/jcorreia/PycharmProjects/BioSystemsUM/ReactEA/src/'
                                                       'reactea/outputs/old_results/'
                                                       'GA_rr_600esc_400gen_0.6sweet_0.4_caloric/'
                                                       'FINAL_TRANSFORMATIONS_04-29_16-51-54.csv'}
        PlotResults(mock_output_configs, solution_index=0).plot_results(save_fig=False)
        PlotResults(mock_output_configs, solution_index=4).plot_results(save_fig=False)
        PlotResults(mock_output_configs, solution_index=6).plot_results(save_fig=False)
