"""
Single-process fine-tuner for Kronos on US equity daily data.

Replaces the DDP + Qlib + comet_ml pipeline in train_predictor.py with
a simple loop that runs on a MacBook (MPS) or any single GPU/CPU.

Usage:
    cd finetune
    python finetune_us.py                   # fine-tunes Kronos-small
    python finetune_us.py --epochs 10       # more epochs
    python finetune_us.py --tickers AAPL MSFT TSLA NVDA  # custom tickers

Saves fine-tuned model to:
    ../finetuned_kronos_us/   (load with Kronos.from_pretrained(...))
"""

import argparse
import os
import sys
import warnings
from typing import Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import yfinance as yf

warnings.filterwarnings("ignore")
sys.path.append("../")
from model.kronos import Kronos, KronosTokenizer

# ── Default ticker universe (diverse US large/mid caps) ─────────────────────
DEFAULT_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","JPM","V","UNH",
    "XOM","JNJ","WMT","MA","PG","HD","BAC","ABBV","CVX","MRK",
    "COST","NFLX","AMD","SOFI","PLTR","COIN","BA","GS","MS","C",
    "WFC","DIS","INTC","MU","QCOM","PYPL","SQ","SHOP","UBER","ABNB",
]

# ── Hyperparameters ──────────────────────────────────────────────────────────
WINDOW     = 101   # lookback(90) + pred_win(10) + 1  (causal LM: predict next token)
NORM_WIN   = 60    # matches inference normalisation in kronos.py
CLIP       = 5.0
BATCH_SIZE = 8
LR         = 2e-5  # small LR for fine-tuning (10× below pre-train)
EPOCHS     = 5
GRAD_CLIP  = 1.0
LOG_EVERY  = 50
SAVE_PATH  = "../finetuned_kronos_us"
TRAIN_END  = "2024-12-31"
VAL_START  = "2024-06-01"


# ── Dataset ──────────────────────────────────────────────────────────────────

def _time_features(ts: pd.Series) -> np.ndarray:
    return np.stack([
        ts.dt.minute.values,
        ts.dt.hour.values,
        ts.dt.weekday.values,
        ts.dt.day.values,
        ts.dt.month.values,
    ], axis=1).astype(np.float32)


def fetch_ticker(ticker: str) -> Optional[pd.DataFrame]:
    try:
        raw = yf.download(ticker, start="2019-01-01", interval="1d",
                          auto_adjust=True, progress=False)
        raw = raw.dropna()
        if len(raw) < WINDOW + 20:
            return None
        df = pd.DataFrame({
            "ts":     raw.index.tz_localize(None) if raw.index.tz else raw.index,
            "open":   raw["Open"].values.flatten(),
            "high":   raw["High"].values.flatten(),
            "low":    raw["Low"].values.flatten(),
            "close":  raw["Close"].values.flatten(),
            "volume": raw["Volume"].values.flatten(),
            "amount": raw["Open"].values.flatten() * raw["Volume"].values.flatten(),
        })
        df["ts"] = pd.to_datetime(df["ts"])
        return df.reset_index(drop=True)
    except Exception as e:
        print(f"  [{ticker}] skip: {e}")
        return None


def make_samples(df: pd.DataFrame, date_end: Optional[str]) -> list:
    feat = ["open", "high", "low", "close", "volume", "amount"]
    mask = pd.Series([True] * len(df))
    if date_end:
        mask &= df["ts"] <= pd.Timestamp(date_end)
    sub = df[mask].reset_index(drop=True)
    if len(sub) < WINDOW:
        return []

    samples = []
    for i in range(len(sub) - WINDOW + 1):
        win  = sub[feat].iloc[i : i + WINDOW].values.astype(np.float32)
        ts_w = sub["ts"].iloc[i : i + WINDOW]

        # normalise using recent NORM_WIN bars of the lookback (no future leakage)
        norm_slice = win[:NORM_WIN]
        mu  = norm_slice.mean(axis=0)
        sig = norm_slice.std(axis=0) + 1e-5
        x   = np.clip((win - mu) / sig, -CLIP, CLIP)

        stamp = _time_features(ts_w)
        samples.append((x, stamp))
    return samples


class USEquityDataset(Dataset):
    def __init__(self, samples: list):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, stamp = self.samples[idx]
        return torch.from_numpy(x), torch.from_numpy(stamp)


# ── Training loop ─────────────────────────────────────────────────────────────

def run_epoch(model, tokenizer, loader, optimizer, scheduler, device, train: bool):
    model.train(train)
    total_loss, n_batches = 0.0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch_x, batch_stamp in loader:
            batch_x     = batch_x.to(device)
            batch_stamp = batch_stamp.to(device)

            with torch.no_grad():
                tokens = tokenizer.encode(batch_x, half=True)

            token_in  = [tokens[0][:, :-1], tokens[1][:, :-1]]
            token_out = [tokens[0][:, 1:],  tokens[1][:, 1:]]

            logits = model(token_in[0], token_in[1], batch_stamp[:, :-1, :])
            loss, _, _ = model.head.compute_loss(
                logits[0], logits[1], token_out[0], token_out[1]
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                if scheduler:
                    scheduler.step()

            total_loss += loss.item()
            n_batches  += 1

    return total_loss / max(n_batches, 1)


def main(tickers: list, epochs: int, batch_size: int) -> None:
    # ── Device ────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────
    print(f"\nFetching {len(tickers)} tickers …")
    train_samples, val_samples = [], []
    for ticker in tqdm(tickers):
        df = fetch_ticker(ticker)
        if df is None:
            continue
        train_samples += make_samples(df, TRAIN_END)
        val_samples   += make_samples(df, None)   # all data, val uses recent portion
        # keep only the val portion that is after VAL_START
        val_df = df[df["ts"] >= VAL_START].reset_index(drop=True)
        if len(val_df) >= WINDOW:
            val_samples += make_samples(val_df, None)

    # deduplicate val (some overlap with train) — just shuffle and cap
    import random; random.seed(42)
    random.shuffle(val_samples)
    val_samples = val_samples[:min(2000, len(val_samples))]

    print(f"Train samples: {len(train_samples):,}   Val samples: {len(val_samples):,}")
    if not train_samples:
        print("No training data. Exiting.")
        return

    train_loader = DataLoader(USEquityDataset(train_samples), batch_size=batch_size,
                               shuffle=True,  num_workers=0, drop_last=True)
    val_loader   = DataLoader(USEquityDataset(val_samples),   batch_size=batch_size,
                               shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────
    print("\nLoading Kronos-small …")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    tokenizer.eval().to(device)
    for p in tokenizer.parameters():
        p.requires_grad = False          # tokenizer is frozen

    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")

    # ── Optimiser ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                   betas=(0.9, 0.95), weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=LR / 10
    )

    # ── Train ─────────────────────────────────────────────────────
    best_val, best_epoch = float("inf"), 0
    os.makedirs(SAVE_PATH, exist_ok=True)

    print(f"\nFine-tuning for {epochs} epochs …\n")
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, tokenizer, train_loader,
                               optimizer, scheduler, device, train=True)
        val_loss   = run_epoch(model, tokenizer, val_loader,
                               None, None, device, train=False)

        flag = ""
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            model.save_pretrained(SAVE_PATH)
            flag = "  ← saved"

        print(f"Epoch {epoch}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}{flag}")

    print(f"\nBest val loss: {best_val:.4f} at epoch {best_epoch}")
    print(f"Fine-tuned model saved to {SAVE_PATH}/")
    print("\nLoad it with:")
    print(f"  model = Kronos.from_pretrained('{SAVE_PATH}')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    parser.add_argument("--epochs",     type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    main(args.tickers, args.epochs, args.batch_size)
