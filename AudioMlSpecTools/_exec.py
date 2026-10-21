###############################################################################
# Global Imports
###############################################################################
import signal
from typing import Optional

###############################################################################
# 3PP Imports
###############################################################################
import click
from playsound3 import playsound

###############################################################################
# Certus Imports
###############################################################################
from .feature_source import FlexibleFeatureChannel, FlexibleFeatureSource
from .vis import plot_with_time_domain
from .wav import load_wav, load_raw_wav, save_wav


###############################################################################
# Config
###############################################################################
KEEP_PLAYING = True


###############################################################################
# Constants
###############################################################################
DEFAULT_SR = 44_100
N_FFT = 1024


###############################################################################
# Helpers
###############################################################################
def _signal_handler(sig, frame):
    global KEEP_PLAYING
    print('You pressed Ctrl+C!')
    KEEP_PLAYING = False


def _get_output_filename(infile: str, suffix: str):
    segs = infile.split(".")
    prefix = ".".join(segs[:-1])
    ext = segs[-1]
    return f"{prefix}-{suffix}.{ext}"


###############################################################################
# ! Commands
###############################################################################
@click.group()
def cli():
    pass


@cli.command("clip")
@click.argument("filename", required=True)
@click.argument("output_filename")
@click.option("--start_sec", "-s", type=int)
@click.option("--end_sec", "-e", type=int)
@click.option("--duration_sec", "-d", type=int)
def clip(filename: str,
         output_filename: Optional[str],
         start_sec: Optional[int],
         end_sec: Optional[int],
         duration_sec: Optional[int],
         ):
    outfile = output_filename or _get_output_filename(filename, "CLIPPED")

    if (start_sec is not None) or (end_sec is not None):
        wav, sr = load_raw_wav(filename, start_sec=start_sec, end_sec=end_sec, mono=False)
    else:
        wav, sr = load_wav(filename, duration_secs=duration_sec, mono=False)

    save_wav(outfile, wav, sr, encoding="PCM_S", bits_per_sample=16)


@cli.command("play")
@click.argument("filename", required=True)
@click.option("--loop", is_flag=True)
def play(filename: str, loop: bool):
    signal.signal(signal.SIGINT, _signal_handler)
    while KEEP_PLAYING and loop:
        playsound(filename)


@cli.command("resample")
@click.argument("filename", required=True)
@click.argument("output_filename")
@click.option("--target_sr", required=True, type=int)
def resample(filename: str,
             output_filename: Optional[str],
             target_sr: int,
             ):
    outfile = output_filename or _get_output_filename(filename, "RESAMPLED")
    wav, _ = load_wav(filename, target_sr=target_sr, mono=False)
    save_wav(outfile, wav, target_sr, encoding="PCM_S", bits_per_sample=16)


@cli.command("show")
@click.argument("filename", required=True)
@click.option("--sr", default=DEFAULT_SR)
@click.option("--duration_sec", "-d", type=int)
def show(filename: str, sr: int, duration_sec: Optional[int]):
    feature_channels = [
        FlexibleFeatureChannel(sr, n_fft=N_FFT, is_logarithmic=True, is_mel=True, scale_spec=True),
        FlexibleFeatureChannel(sr, n_fft=N_FFT, is_logarithmic=True, is_mel=False, scale_spec=True),
    ]
    source = FlexibleFeatureSource(feature_channels)

    wav, _ = load_wav(filename, target_sr=sr, duration_secs=duration_sec)

    spectrum = source.forward(wav)

    if spectrum.ndim > 3:
        spectrum = spectrum.squeeze(0)

    plot_with_time_domain(spectrum, wav, sr, filename.split("/")[-1])


###############################################################################
# ! Main
###############################################################################
if __name__ == "__main__":
    cli()
