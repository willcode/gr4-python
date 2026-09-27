"""Writes a tone to a file through a head block, then reads the file back.

    python3 tone_to_file.py [path]

Without a path the file is written into a temporary directory, which is removed at the end.
"""

import array
import os
import sys
import tempfile

import gr4

SAMPLE_RATE = 48_000
SAMPLES = 4_800


def write_tone(path: str) -> None:
    """Runs a tone of 440 Hz through a head block of SAMPLES samples into a file sink writing to path."""
    graph = gr4.Graph()
    tone = graph.add(
        "gr::blocks::basic::SignalGenerator<float32>",
        "tone",
        sample_rate=SAMPLE_RATE,
        tone_frequency=440,
        amplitude=0.5,
    )
    head = graph.add(gr4.registry.find("testing::HeadBlock", "float32"), "head", n_samples_max=SAMPLES)
    sink = graph.add(gr4.registry.find("fileio::BasicFileSink", "float32"), "sink", file_name=path)
    graph.connect(tone, head, sink)
    graph.run()


def read_samples(path: str) -> array.array:
    """Returns the float32 samples of a raw file."""
    samples = array.array("f")
    with open(path, "rb") as file:
        samples.frombytes(file.read())
    return samples


def main() -> None:
    """Writes the tone, reads it back and prints what the file holds."""
    with tempfile.TemporaryDirectory() as directory:
        # The file sink takes a path with a directory part.
        path = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(directory, "tone.f32")
        write_tone(path)
        samples = read_samples(path)
        print(f"{len(samples)} samples, {os.path.getsize(path)} bytes in {os.path.basename(path)}")
        print(f"peak {max(abs(sample) for sample in samples):.4f}")
        print("first samples", " ".join(f"{sample:+.4f}" for sample in samples[:6]))


if __name__ == "__main__":
    main()
