import unittest
import os
from unittest.mock import patch, MagicMock
import logging

# from dslpipe.pipeline.pipeline import Manager
from dslflagger.tasks.example_rfi_flag import RFIFlag

class TestRFIFlag(unittest.TestCase):

    def setUp(self):
        # Set TL_OUTPUT environment variable for testing
        os.environ['TL_OUTPUT'] = '/tmp/test_output'

        # Mock mpiutil.rank0 to be True for testing logging
        self.mock_mpiutil_rank0 = patch('caput.mpiutil.rank0')
        self.mock_mpiutil_rank0_obj = self.mock_mpiutil_rank0.start()
        self.mock_mpiutil_rank0_obj.return_value = True

        # Mock the module-level logger in example_rfi_flag
        self.mock_logger = MagicMock()
        self.patch_logger = patch('dslflagger.tasks.example_rfi_flag.logger', new=self.mock_logger)
        self.patch_logger.start()

        # Mock OneAndOne.__init__ to prevent complex initialization
        self.patch_one_and_one_init = patch('dslpipe.pipeline.pipeline.OneAndOne.__init__', return_value=None)
        self.patch_one_and_one_init.start()

    def tearDown(self):
        # Clean up environment variable
        del os.environ['TL_OUTPUT']
        self.mock_mpiutil_rank0.stop()
        self.patch_logger.stop()
        self.patch_one_and_one_init.stop()

    def test_process_method(self):
        # Test with default algorithm
        rfi_flag_task = RFIFlag(parameter_file_or_dict={'algorithm': 'algorithm1'})
        rfi_flag_task.params = {'algorithm': 'algorithm1'} # Manually set params as __init__ is mocked
        rfi_flag_task.process(input='test_input')
        self.mock_logger.info.assert_called_with('Executing RFI flagging task with algorithm: algorithm1')

        # Test with a different algorithm
        rfi_flag_task = RFIFlag(parameter_file_or_dict={'algorithm': 'algorithm2'})
        rfi_flag_task.params = {'algorithm': 'algorithm2'} # Manually set params as __init__ is mocked
        rfi_flag_task.process(input='test_input')
        self.mock_logger.info.assert_called_with('Executing RFI flagging task with algorithm: algorithm2')

    def test_read_input_method(self):
        mock_input_files = ['file1.h5', 'file2.h5']
        rfi_flag_task = RFIFlag(parameter_file_or_dict={'input_files': mock_input_files})
        rfi_flag_task.params = {'input_files': mock_input_files} # Manually set params as __init__ is mocked
        self.assertEqual(rfi_flag_task.read_input(), mock_input_files)

if __name__ == '__main__':
    unittest.main()