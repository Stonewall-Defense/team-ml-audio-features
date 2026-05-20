# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
