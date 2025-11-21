# 🧠 Audio Super‑Resolution (SBE: CNN+GAN) — Project Board

_Updated: 2025-10-18_  
Use this as your **Kanban inside VS Code**. Open it alongside the repo and keep checking boxes as you go.

> Tip: Install **Todo Tree** and **Markdown All in One**. Pin this file in VS Code. Use `Ctrl+Shift+V` for preview.

---

## 🗂 Columns
- **Backlog** → ideas and future work
- **This Week** → focus items for the current week
- **Today** → 3–5 concrete tasks
- **In Progress** → currently active
- **Blocked** → waiting on data/hardware/answers
- **Done** → completed

---

## 🚀 This Week (Sprint Goal)
- [ ] Baseline: prepare NB/WB pairs, run interpolation baseline, compute PESQ/STOI
- [ ] Train first CNN+GAN for 5–10 epochs (sanity check)
- [ ] Set up debugger breakpoints + tensor shape watches
- [ ] Make 1 notebook: visualize spectrograms & mel for NB vs WB
- [ ] Push results & notes to GitHub (README update + figures)

---

## ✅ Today (Top 3–5)
- [ ] Verify `.venv` active + interpreter selected in VS Code
- [ ] Run `prepare_pairs.py` on 5–10 WAVs
- [ ] Run baseline + evaluation (produce `metrics.csv`)
- [ ] Create `notebooks/01_audio_explore.ipynb` and add 3 plots (waveform, STFT, mel)
- [ ] Commit & push

---

## 🏗 In Progress
- [ ] Implement GAN training (PatchGAN + L1) — `src/trainers/train_cnn_gan.py`
- [ ] Add preLaunchTask to run data prep before train — `.vscode/launch.json`

---

## ⛔ Blocked
- [ ] Need more 16 kHz clean speech data (WB) for training set
- [ ] (Optional) GPU access for faster epochs

---

## 📦 Backlog (Ideas / Later)
- [ ] Add Griffin–Lim (quick vocoder) for mel → audio listening tests
- [ ] Switch from mel-space to complex STFT with magnitude + phase estimation
- [ ] Try HiFi-GAN or UnivNet vocoder for higher fidelity
- [ ] Add data augmentation: SNR, small EQ, time-stretch
- [ ] Compare GAN vs simple L2 CNN (no adversarial loss) ablation
- [ ] Add MOS A/B page hosting via GitHub Pages
- [ ] Export best model + small demo CLI tool
- [ ] Create LaTeX report skeleton and slide deck

---

## 📚 Phase Checklist (Roadmap)
### Phase 0 — Setup
- [ ] VS Code: `.vscode/` configured (settings/launch/tasks/extensions)
- [ ] GitHub: remote connected, branching tested
- [ ] Baseline & eval run once

### Phase 1 — Audio/DSP
- [ ] NB vs WB exploration (plots + notes)
- [ ] LPF + down/up sampling validated
- [ ] PESQ/STOI pipeline sanity check

### Phase 2 — CNNs
- [ ] Understand conv/padding/stride; draw U‑Net diagram
- [ ] Train simple CNN (no GAN) to predict WB mel
- [ ] Visualize intermediate feature maps

### Phase 3 — GANs
- [ ] Add discriminator (PatchGAN) and adv loss
- [ ] Stabilize training (learning rates, betas, lambda_l1)
- [ ] Save/inspect samples per epoch

### Phase 4 — Evaluation
- [ ] Compute PESQ/STOI charts (baseline vs CNN vs GAN)
- [ ] Run small MOS A/B test with 5–10 listeners
- [ ] (Optional) DRT on minimal pairs

### Phase 5 — Git/Docs
- [ ] Branches for features; tidy commit messages
- [ ] README: quickstart + results
- [ ] Tag release v0.1 (baseline), v0.2 (GAN), v1.0 (final)

### Phase 6 — Final Deliverables
- [ ] Report (LaTeX) with figures/tables
- [ ] Slide deck with audio demos
- [ ] Code cleanup + instructions

---

## 📝 Daily Log (fill as you go)
- **2025-10-20** —
  - What I planned: 
  - What I did:
  - Roadblocks:
  - Next actions:
  - Notes 
    -  speech dataset from https://openslr.org/12/
    -  Music dataset from - https://www.kaggle.com/datasets/ayu055/musdb18hq?resource=download

---

## 🔧 Useful Commands (copy/paste)
```bash
# venv + deps
python -m venv .venv
# Win: .venv\Scripts\activate    mac/Linux: source .venv/bin/activate
pip install -r requirements.txt

# data prep
python src/utils/prepare_pairs.py --in_dir data/raw_wb --out_dir data/processed_nbwb --sr_wb 16000 --sr_nb 8000

# baseline
python src/baselines/run_baseline.py --pairs data/processed_nbwb/manifest.csv --out_dir experiments/baseline_out
python src/eval/eval_metrics.py --pairs_csv data/processed_nbwb/manifest.csv --pred_dir experiments/baseline_out --sr 16000

# training
python src/trainers/train_cnn_gan.py --config configs/cnn_gan_mel.yaml
```
