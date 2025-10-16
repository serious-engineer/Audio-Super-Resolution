import os, argparse, pandas as pd, soundfile as sf, numpy as np, librosa
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="CSV manifest from prepare_pairs.py")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.pairs)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        y_nb, sr = sf.read(row["nb_path"])
        if y_nb.ndim > 1: y_nb = y_nb.mean(axis=1)
        # upsample NB from 8k→16k with linear interpolation (baseline)
        y_up = librosa.resample(y_nb, orig_sr=int(row["sr_nb"]), target_sr=int(row["sr_wb"]))
        out_path = os.path.join(args.out_dir, os.path.basename(row["wb_path"]).replace(".wav", "_baseline.wav"))
        sf.write(out_path, y_up, int(row["sr_wb"]))
    print("Baseline outputs saved to", args.out_dir)

if __name__ == "__main__":
    main()
