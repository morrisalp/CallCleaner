# CallCleaner

**Lightweight bioacoustic denoising from an audio foundation model.**

CallCleaner removes background noise (wind, rain, vehicles, other animals) from recordings of animal
vocalizations. It is a 19M-parameter model distilled from the source-separation foundation model
SAM-Audio; it needs no detector, text prompt or GPU, and runs about 90× faster than real time on a
laptop CPU.

[Project page](https://callcleaner.github.io) · [Model (Hugging Face)](https://huggingface.co/malper/CallCleaner) · [Paper (arXiv)](https://arxiv.org/abs/2611.11491)

## Install

Python 3.11:

```bash
git clone https://github.com/morrisalp/CallCleaner && cd CallCleaner
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python denoise.py noisy.wav out/                 # one file -> out/noisy.wav
python denoise.py recordings/ out/               # a folder (recursively)
python denoise.py recordings/ out/ --keep-sr     # write at the input's sample rate
```

The model works at 16 kHz mono: input is downmixed and resampled, and long files are processed in
overlapping 60 s chunks. Content above 8 kHz is not restored. Weights are downloaded from
[Hugging Face](https://huggingface.co/malper/CallCleaner) on first use; the model card also shows how
to call the model from Python.

Evaluation code (our metrics on the Biodenoising benchmark) is coming soon.

## License

Code: MIT ([LICENSE](LICENSE)). Model weights: SAM License (see the
[model card](https://huggingface.co/malper/CallCleaner)), as they are distilled from SAM-Audio. The
model architecture is from [Biodenoising](https://github.com/earthspecies/biodenoising) (MIT).

## Citation

```bibtex
@article{xie2026callcleaner,
  title         = {CallCleaner: Lightweight Bioacoustic Denoising from an Audio Foundation Model},
  author        = {Michael Xie and Ga{\v{s}}per Begu{\v{s}} and Morris Alper},
  year          = {2026},
  eprint        = {2611.11491},
  archivePrefix = {arXiv}
}
```
