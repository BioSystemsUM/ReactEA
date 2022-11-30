import os
import shutil
from unittest import TestCase

from tests import TEST_DIR


class ReactEACLITestCase(TestCase):

    def setUp(self):
        self.output_folder = os.path.join(TEST_DIR, 'data/output/')
        self.config_file = os.path.join(TEST_DIR, 'configs/base_config.yaml')

    def tearDown(self):
        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)


class TestBioCatalyzerCLI(ReactEACLITestCase, TestCase):

    def test_reactea_cli(self):
        exit_status = os.system('reactea --help')
        self.assertEqual(exit_status, 0)

    def test_reactea_cli_missing_args(self):
        # missing argument 'COMPOUNDS'
        exit_status = os.system('reactea')
        self.assertEqual(exit_status, 512)

        # missing argument 'OUTPUT_PATH'
        exit_status = os.system('reactea dummy_arg_1')
        self.assertEqual(exit_status, 512)

    def test_reactea_cli_working(self):
        exit_status = os.system(f"reactea {self.config_file}")
        self.assertEqual(exit_status, 0)
