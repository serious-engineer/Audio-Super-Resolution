import os, argparse, glob, pandas as pd, numpy as np, soundfile as sf, librosa, scipy.signal as sig

def lowpass(y, sr, cutoff):
    ny = sr/2
    Wn = cutoff/ny
    b, a = sig.butter(8, Wn, btype='low')
    return sig.filtfilt(b, a, y)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Folder with WB .wav at target sr_wb (mono preferred)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sr_wb", type=int, default=16000)
    ap.add_argument("--sr_nb", type=int, default=8000)
    ap.add_argument("--cutoff", type=float, default=3800.0, help="LPF cutoff for NB creation (Hz)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for wav in glob.glob(os.path.join(args.in_dir, "*.wav")):
        y, sr = sf.read(wav)
        if y.ndim > 1: y = y.mean(axis=1)
        if sr != args.sr_wb:
            y = librosa.resample(y, orig_sr=sr, target_sr=args.sr_wb)
        # Create NB by lowpass + downsample to 8k
        y_lp = lowpass(y, args.sr_wb, args.cutoff)
        y_nb = librosa.resample(y_lp, orig_sr=args.sr_wb, target_sr=args.sr_nb)
        # Upsample NB back to 16k for model input/baseline
        y_nb_ups = librosa.resample(y_nb, orig_sr=args.sr_nb, target_sr=args.sr_wb)

        base = os.path.splitext(os.path.basename(wav))[0]
        wb_path = os.path.join(args.out_dir, f"{base}_WB.wav")
        nb_path = os.path.join(args.out_dir, f"{base}_NB.wav")
        nb_ups_path = os.path.join(args.out_dir, f"{base}_NBups.wav")

        sf.write(wb_path, y, args.sr_wb)
        sf.write(nb_path, y_nb, args.sr_nb)
        sf.write(nb_ups_path, y_nb_ups, args.sr_wb)

        rows.append(dict(wb_path=wb_path, nb_path=nb_path, nb_ups_path=nb_ups_path, sr_wb=args.sr_wb, sr_nb=args.sr_nb))

    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "manifest.csv"), index=False)
    print(f"Wrote {len(rows)} pairs and manifest.csv to {args.out_dir}")

if __name__ == "__main__":
    main()
