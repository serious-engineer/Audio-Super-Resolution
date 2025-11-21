python -m src.utils.make_low_sr_dataset  --data_root data  --sample_rate 48000  --low_sr 8000

python -m src.train  --dataset_dir data  --out_dir checkpoints  --sample_rate 48000  --batch_size 32  --epochs 100
python -m src.train  --dataset_dir data  --out_dir checkpoints  --sample_rate 48000  --batch_size 32  --epochs 300  --resume


python -m src.infer --generator_ckpt checkpoints/generator_final.pt --hr_dir data/hr --output_dir outputs
python -m src.infer --generator_ckpt checkpoints/generator_final.pt --hr_dir data/hr --max_files 10 --output_dir outputs
python -m src.infer --input_wav data/low_sr/p225_001.wav --generator_ckpt checkpoints/generator_final.pt --hr_dir data/hr --output_dir outputs
python -m src.infer --generator_ckpt checkpoints/generator_epoch_500.pt --low_sr_dir data/low_sr --hr_dir data/hr  --output_dir outputs_hifigan --vocoder hifigan48 --max_files 10

python -m src.infer --generator_ckpt checkpoints/generator_final.pt --input_wav data/low_sr/p225_001.wav --vocoder vocos48 --sample_rate 48000 --n_mels 128 --hr_dir data/hr --output_dir outputs

python -m src.infer --generator_ckpt checkpoints/generator_final.pt --input_wav data/low_sr/p225_001.wav --vocoder hifigan48 --sample_rate 48000 --n_mels 128 --hr_dir data/hr --output_dir outputs

python -m src.eval --hr_dir data/hr --sr_dir outputs --sample_rate 48000

python train.py -c config_48k.json -m hifigan_48k

python -m src.utils.vocoder_eval --hr_dir data/test/ --out_dir test_out --vocoder hifigan4 --max_files 5 --save_recon 

# AudioSR-GAN Workspace

This workspace implements a minimal SRGAN-style audio super-resolution model
inspired by the AudioSR paper. The model operates in **mel-spectrogram space**:

- Input:  log-mel spectrogram of band-limited audio (e.g., 8 kHz upsampled to 48 kHz)
- Output: log-mel spectrogram with hallucinated high-frequency content
- Backbone: SRResNet-style generator + discriminator adapted from SRGAN

## Layout

```text
audiosr_gan_workspace/
  src/
    models/
      audio_srgan.py       # Generator + discriminator on mel-spectrograms
    data/
      audio_dataset.py     # Dataset that builds (low, high) mel pairs from WAVs
    utils/
      mel_utils.py         # Log-mel extractor and helpers
    train.py               # Training script for the GAN
    infer.py               # Simple inference script that outputs super-res mel
  configs/
  requirements.txt
  README.md
```

## Quickstart

1. Create a virtual environment and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Prepare a folder of WAV files at your target sample rate (default 48 kHz).

3. Train the AudioSR GAN:

   ```bash
   python -m src.train \\
       --dataset_dir /path/to/wavs \\
       --out_dir checkpoints \\
       --sample_rate 48000 \\
       --low_sr 8000 \\
       --batch_size 8 \\
       --epochs 5
   ```

4. Run inference to get a super-resolved mel-spectrogram:

   ```bash
   python -m src.infer \\
       --input_wav path/to/example.wav \\
       --generator_ckpt checkpoints/generator_final.pt \\
       --output_mel mel_sr.pt
   ```

> **Note:** To reconstruct waveforms from `mel_sr.pt`, you need to plug in a
> neural vocoder such as HiFi-GAN that takes log-mel as input. This workspace
> focuses on the super-resolution model itself and leaves the vocoder integration
> as a TODO.
