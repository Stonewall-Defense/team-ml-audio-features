# Test Imports
###############################################################################
import unittest

from AudioMlSpecTools.wav import WavReader


###############################################################################
# Config
###############################################################################
AUDIO_FILE = "test/res/380ACP-7-7WYYO4zK0hPS-9.wav"


###############################################################################
# Tests
###############################################################################
class TestWavReader(unittest.TestCase):
    def test_pad(self):
        reader = WavReader(pad=True)
        audio = reader.read(AUDIO_FILE, end_sec=1.0)
        self.assertEqual(audio.shape[-1], 44_100)

    def test_resample_one_off(self):
        reader = WavReader(target_sr=22_050, pad=True)
        audio = reader.read(AUDIO_FILE, end_sec=1.0)
        self.assertEqual(audio.shape[-1], 22_050)

    def test_resample_no_pad(self):
        reader = WavReader(target_sr=22_050)
        audio = reader.read(AUDIO_FILE, end_sec=1.0)
        self.assertEqual(audio.shape[-1], 10_130)

    def test_resample_cached(self):
        reader = WavReader(target_sr=22_050, expected_file_sr=44_100, pad=True)
        audio = reader.read(AUDIO_FILE, end_sec=1.0)
        self.assertEqual(audio.shape[-1], 22_050)

    def test_resample_raise(self):
        reader = WavReader(target_sr=22_050, expected_file_sr=48_000)
        with self.assertRaises(ValueError):
            reader.read(AUDIO_FILE)
