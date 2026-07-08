# AGENTS.md

## What this repo does

Extracts wave samples from SoundFont2 (`.sf2`) files, applies signal processing (resampling, FFT band-limiting, loop-point fixing, gain), and generates C/Python wavetable code for embedded MCU targets (8051, AVR, STM8, generic).

## Entrypoint

- `WaveTableGenerator.py` — CLI tool, reads `--sf2`, applies pipeline, renders templates
- `Mapping.py` — standalone utility: prints a 10-bit audio → 16-bit PWM lookup table as C array

## Dependencies

```
pip install numpy scipy matplotlib sf2utils
```

## Key CLI options

```bash
# List samples in an SF2
python WaveTableGenerator.py --sf2 soundfont/MusicBox.sf2 --listSf2

# Generate wavetable code for a specific sample
python WaveTableGenerator.py --sf2 soundfont/MusicBox.sf2 --template 8051_sdcc \
    --sampleName "Square Wave C5" --outSampleRate 32000 --outSampleWidth 1 \
    --lowestNote 36 --padding --outputDir ./out

# Generate spectrum/time-domain/loop-error PDF for debugging
python WaveTableGenerator.py --sf2 soundfont/MusicBox.sf2 --template generic \
    --sampleName "Music Box C5" --spectrumPdf analysis.pdf
```

Template dirs: `template/{8051_sdcc,avr_gcc,generic,python,stm8_sdcc}/`. Each folder's `.template` files are rendered via `string.Template.safe_substitute`. The `--extraTemplate` flag overrides built-in templates with custom file paths.

## Processing pipeline

1. Extract attack + loop samples from SF2
2. Resample/convert channel/sample width to target
3. Estimate fundamental frequency via FFT
4. FFT band-limit to prevent aliasing at `--lowestNote` (default MIDI 36)
5. Auto-fix loop point discontinuity (minimizes value + slope error)
6. Apply gain in dB (`--gainDb`, e.g. -6.0)
7. Render template files with sample data + pre-computed increment table
8. Optionally output a `--spectrumPdf` with before/after plots

## Architecture notes

- `genCode()` populates `$WaveTable*` template variables — see template files for the full param list
- `getFromSf2()` reads raw SF2 sample data using `sf2utils`, returns `(name, midiNote, attack, loop, width, rate, channels)`
- `readWaveSamples()` reads legacy `.wav` files (mono, 16-bit, 32000 Hz only) — not used in the SF2 pipeline
- `Mapping.py` generates a `const uint16_t audio2pwm_lut[1024]` C array; run it directly: `python Mapping.py`

## Conventions

- `.venv/` in `.gitignore`; dependencies: numpy, scipy, matplotlib, sf2utils
- No test framework set up; manual testing via CLI
- SF2 file at `soundfont/MusicBox.sf2`; legacy WAV samples in `legacyWaveSamples/`
- License: LGPL-3.0
