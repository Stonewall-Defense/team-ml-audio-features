# Audio Features

Convenience functions for generating ML features from audio data.

## Prerequisites

- Python 3.11 runtime
- Pip for package installation

## Installation

Install the dependencies into the environment with [pip](https://pypi.org/project/pip/):

```bash
pip install -r requirements.txt
```

Then install the package itself locally:

```bash
pip install .
```

## Testing

```bash
python3 -m coverage run -m unittest discover -s test -p "*_test.py" && python -m coverage report --skip-covered
python -m coverage html
```

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/Stonewall-Defense/team-ml-audio-features/tags).

## Authors

- **Ryan Quinn** - *Initial work*

## License

MIT.
