from unittest.mock import patch

from src.utils.device import setup_device


class TestSetupDevice:
    @patch("torch.cuda.is_available")
    def test_setup_device_cuda_available(self, mock_is_available):
        # Mock torch.cuda.is_available to return True
        mock_is_available.return_value = True

        device = setup_device()

        # Assert that the device is set to CUDA
        assert device.type == "cuda"

    @patch("torch.cuda.is_available")
    def test_setup_device_cuda_not_available(self, mock_is_available):
        # Mock torch.cuda.is_available to return False
        mock_is_available.return_value = False

        device = setup_device()

        # Assert that the device is set to CPU
        assert device.type == "cpu"
