"""
Formant Shift Augmentation using Praat (via Parselmouth)

Requirements:
    pip install numpy soundfile parselmouth-praat

Usage:
    python formant_shift_praat.py --input input.wav --output output.wav --formant-ratio 1.10
"""

import argparse

import numpy as np
import soundfile as sf
import parselmouth
from parselmouth.praat import call


def formant_shift_praat(
    in_wav: str,
    out_wav: str,
    formant_ratio: float = 1.10,
    pitch_ratio: float = 1.00,
    duration_ratio: float = 1.00,
    f0_min: float = 75.0,
    f0_max: float = 500.0,
) -> str:
    """Shift formant frequencies using Praat's 'Change gender...' command.

    Args:
        in_wav:         Path to input WAV file.
        out_wav:        Path to output WAV file.
        formant_ratio:  Formant shift ratio (>1.0 = up, <1.0 = down).
        pitch_ratio:    Pitch shift ratio (1.0 = unchanged).
        duration_ratio: Duration ratio (1.0 = unchanged).
        f0_min:         Minimum F0 for pitch analysis (Hz).
        f0_max:         Maximum F0 for pitch analysis (Hz).

    Returns:
        Path to the saved output WAV file.
    """
    snd = parselmouth.Sound(in_wav)

    # Estimate original pitch median
    pitch = call(snd, "To Pitch", 0.0, f0_min, f0_max)
    f0_values = pitch.selected_array["frequency"]
    f0_values = f0_values[(f0_values > 0) & np.isfinite(f0_values)]
    orig_pitch_median = float(np.median(f0_values)) if len(f0_values) else 200.0
    new_pitch_median = orig_pitch_median * pitch_ratio

    # Apply formant shift via Praat
    shifted = call(
        snd,
        "Change gender...",
        f0_min,
        f0_max,
        formant_ratio,
        new_pitch_median,
        1.0,              # pitch_range_factor
        duration_ratio,
    )

    # Convert to numpy and save
    y = shifted.values.T
    if y.ndim == 2 and y.shape[1] > 1:
        y = y.mean(axis=1)  # stereo → mono
    else:
        y = y.squeeze()

    sf.write(out_wav, y, int(shifted.sampling_frequency))
    return out_wav


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Formant shift augmentation (Praat)")
    parser.add_argument("--input",  "-i", required=True, help="Input WAV path")
    parser.add_argument("--output", "-o", required=True, help="Output WAV path")
    parser.add_argument("--formant-ratio",  type=float, default=1.10, help="Formant shift ratio (default: 1.10)")
    parser.add_argument("--pitch-ratio",    type=float, default=1.00, help="Pitch shift ratio (default: 1.00)")
    parser.add_argument("--duration-ratio", type=float, default=1.00, help="Duration ratio (default: 1.00)")
    parser.add_argument("--f0-min", type=float, default=75.0,  help="Min F0 in Hz (default: 75)")
    parser.add_argument("--f0-max", type=float, default=500.0, help="Max F0 in Hz (default: 500)")
    args = parser.parse_args()

    result = formant_shift_praat(
        in_wav=args.input,
        out_wav=args.output,
        formant_ratio=args.formant_ratio,
        pitch_ratio=args.pitch_ratio,
        duration_ratio=args.duration_ratio,
        f0_min=args.f0_min,
        f0_max=args.f0_max,
    )
    print(f"Saved to: {result}")
