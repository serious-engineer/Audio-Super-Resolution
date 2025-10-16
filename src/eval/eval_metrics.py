import os, argparse, pandas as pd, soundfile as sf, numpy as np
from pesq import pesq
from pystoi import stoi
from tqdm import tqdm

def time_align(ref, deg):
    # naive align: pad/crop to same length
    n = min(len(ref), len(deg))
    return ref[:n], deg[:n]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_csv", required=True)
    ap.add_argument("--pred_dir", required=True, help="Dir with *_baseline.wav or model outputs")
    ap.add_argument("--sr", type=int, default=16000)
    args = ap.parse_args()

    df = pd.read_csv(args.pairs_csv)
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        ref_path = row["wb_path"]
        pred_name = os.path.basename(ref_path).replace(".wav", "_baseline.wav")
        pred_path = os.path.join(args.pred_dir, pred_name)
        if not os.path.exists(pred_path): continue
        ref, sr = sf.read(ref_path); pred, sr2 = sf.read(pred_path)
        if ref.ndim>1: ref = ref.mean(axis=1)
        if pred.ndim>1: pred = pred.mean(axis=1)
        ref, pred = time_align(ref, pred)
        # PESQ NB/WB: for 16 kHz use wb mode
        pesq_wb = pesq(args.sr, ref, pred, 'wb')
        stoi_v = stoi(ref, pred, args.sr, extended=False)
        rows.append(dict(file=pred_name, PESQ_WB=pesq_wb, STOI=stoi_v))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.pred_dir, "metrics.csv"), index=False)
    print(out.describe())

if __name__ == "__main__":
    main()
