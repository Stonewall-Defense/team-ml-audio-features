# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.15.0] - 2026-08-07

### Added

- `LowPassFilter` preprocessor option
- Common `OneSidedFilter` parent class for preprocessor filters

### Fixed

- Properly read samples instead of "frames" from multichannel WAV files

## [0.14.0] - 2026-08-06

### Added

- `nan_to_zero` option to replace `nan` values from processing empty audio files with `0`

## [0.13.0] - 2026-08-04

### Changed

- Speed up WAV file read by 10x

## [0.11.0] - 2026-07-10

### Added

- Cached `Resample` object

### Changed

- Moved resampling code to own module
- Allow `float` durations for loading WAV files
- Reworked `WavReader` class for broader applicability

### Removed

- `load_wav_as_is` rolled into new `load_wav` function

## [0.10.0] - 2026-07-09

### Added

- Exposes mel functions

## [0.9.0] - 2026-06-29

### Changed

- `save_wav` function can now take `str` and `int` params for format/bits

## [0.8.0] - 2026-06-12

### Added

- More audio file loading options
- `save_wav` function inspired by `torchaudio`
- Enumerated types for audio formats and bits per sample

### Changed

- Exposed some internal audio convenience functions like `is_multichannel` and `to_mono`

### Removed

- `requirements.txt` file (install dependencies from `pyproject.toml` now)

## [0.7.0] - 2026-06-05

### Changed

- `WavReader` now loads files incrementally

## [0.6.0] - 2026-05-20

### Added

- Selection of FFT window functions
- Automatic `hop_len` determination from window function
- Spectrogram-domain `AudioPostprocessor` classes and associated logic
- `CHANGELOG.md` file!

### Changed

- Break up library functions into logical components
- Variable `hop_len` is now set in class `ExportableSTFT` instead of the feature source classes
  - `hop_len` is now a `kwarg`
- Factored consistent pre/post processor logic into parent class(es)
- Moved abstract methods in `_util.py` base classes from `__call__` to an internal function

### Fixed

- Properly set dependencies in `pyproject.toml`
- Properly set dependencies in `requirements.txt`
