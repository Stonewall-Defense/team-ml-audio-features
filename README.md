# Audio Features

Convenience functions for generating ML features from audio data. Breaks audio ML dependencies on `torchaudio`. Unlike `pytorch` features, these functions can be exported to ExecuTorch and ONNX with no issues.

## Motivation

Except in specific circumstances like `wav2vec`, raw audio has proven to be a much worse input for ML models than spectrogram-based features across a wide variety of problem domains, including environmental sound classificarion ([Guzhov et al. (2021)](https://arxiv.org/pdf/2104.11587)), singing technique classification ([Yamamoto et al. (2021)](https://www.slis.tsukuba.ac.jp/lspc/0000890.pdf)), and ship classification ([Xie, Ren, and Xu (2024)](https://arxiv.org/pdf/2306.01002)).

There is no scientific consensus on the relative benefits of mel-scale spectrograms, linear spectrograms, and MFCCs. Different researchers have shown good results with each type of spectrogram; see respectively [Raponi, Oligeri, and Ali (2021)](https://arxiv.org/pdf/2004.07948), [Jung at al. (2021)](https://www.mdpi.com/2075-4418/11/4/732), and [Razani et al (2017)](https://www.ece.mcgill.ca/~bchamp/Papers/Conference/ISSPIT2017.pdf).

With this library, you can easily try as many feature extraction methods as you want to see what works for your use case.

## Prerequisites

- Python 3.12 runtime
- `pip` for package installation
- Note that `torchcodec` depends on a system installation of FFmpeg

## Installation

Install the dependencies into the environment with [pip](https://pypi.org/project/pip/):

```bash
pip install -r requirements.txt
```

Then install the package itself locally:

```bash
pip install .
```

## Usage

See `examples/features.py`.

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
