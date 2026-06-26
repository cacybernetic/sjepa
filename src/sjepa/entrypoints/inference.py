"""Standalone ONNX inference for the S-JEPA encoder.

Usage:
    infersjepa -c cpu/configs/export.yaml
    infersjepa -c cpu/configs/export.yaml --audio path/to/clip.wav

This script is fully self-contained on purpose. It imports only numpy,
soundfile, onnxruntime, and pyyaml, so you can copy it into another project and
run it without the rest of this code base. It loads the ONNX encoder, reads an
audio clip, turns it into a mono 16 kHz waveform, and prints the frame features.
"""

from __future__ import annotations

import argparse

import numpy as np
import onnxruntime as ort
import soundfile as sf
import yaml

TARGET_RATE = 16000


def load_audio(path, target_rate=TARGET_RATE):
    """Read an audio file as a mono float32 waveform at the target rate."""
    data, source_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if source_rate != target_rate:
        mono = resample_linear(mono, source_rate, target_rate)
    return mono.astype(np.float32)


def resample_linear(signal, source_rate, target_rate):
    """Resample a 1D signal with simple linear interpolation."""
    duration = signal.shape[0] / float(source_rate)
    target_len = int(round(duration * target_rate))
    if target_len <= 1:
        return signal
    source_x = np.linspace(0.0, 1.0, num=signal.shape[0], endpoint=False)
    target_x = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
    return np.interp(target_x, source_x, signal)


def run_encoder(model_path, waveform):
    """Run the ONNX encoder on one waveform and return frame features."""
    session = ort.InferenceSession(model_path,
                                   providers=["CPUExecutionProvider"])
    batch = waveform.reshape(1, 1, -1).astype(np.float32)
    outputs = session.run(["features"], {"waveform": batch})
    return outputs[0]


def _read_config(config_path):
    """Read the YAML config and return its dict."""
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _pick_audio(config, override):
    """Choose the audio path from the override or the config."""
    if override:
        return override
    audio = config.get("audio")
    if not audio:
        raise ValueError("no audio path given; use --audio or set 'audio'")
    return audio


def main():
    """Console entry point for the infersjepa command."""
    parser = argparse.ArgumentParser(description="Run S-JEPA ONNX inference")
    parser.add_argument("-c", "--config", required=True,
                        help="YAML config with 'onnx_path' and 'audio'")
    parser.add_argument("--audio", default=None, help="audio file to encode")
    args = parser.parse_args()
    config = _read_config(args.config)
    model_path = config.get("onnx_path", "model.onnx")
    audio_path = _pick_audio(config, args.audio)
    waveform = load_audio(audio_path)
    features = run_encoder(model_path, waveform)
    print(f"features shape: {features.shape}")
    print(f"feature mean: {float(features.mean()):.4f} "
          f"std: {float(features.std()):.4f}")


if __name__ == "__main__":
    main()
