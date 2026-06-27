#!/usr/bin/env python3
"""Plot matrix multiplication benchmark results."""

import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt


def load_results(files):
    dfs = []
    for f in files:
        if os.path.exists(f):
            df = pd.read_csv(f)
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError("No result files found")
    return pd.concat(dfs, ignore_index=True)


def plot_metric(df, metric, ylabel, output_path):
    plt.figure(figsize=(10, 6))
    for name, group in df.groupby("name"):
        plt.plot(group["m"], group[metric], marker="o", label=name)
    plt.xlabel("Matrix size (M=K=N)")
    plt.ylabel(ylabel)
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_speedup(df, baseline_name, output_path):
    baseline = df[df["name"] == baseline_name].set_index("m")["median_ms"]
    plt.figure(figsize=(10, 6))
    for name, group in df.groupby("name"):
        if name == baseline_name:
            continue
        merged = group.merge(baseline.rename("baseline"), left_on="m", right_index=True)
        speedup = merged["baseline"] / merged["median_ms"]
        plt.plot(merged["m"], speedup, marker="o", label=name)
    plt.xlabel("Matrix size (M=K=N)")
    plt.ylabel(f"Speedup vs {baseline_name}")
    plt.xscale("log", base=2)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument("--input", nargs="+", default=["results/cpu_results.csv", "results/gpu_results.csv", "results/python_results.csv"])
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--baseline", default="cpu_naive", help="Baseline for speedup plot")
    args = parser.parse_args()

    df = load_results(args.input)
    os.makedirs(args.output_dir, exist_ok=True)

    plot_metric(df, "median_ms", "Median latency (ms)", os.path.join(args.output_dir, "latency.png"))
    plot_metric(df, "gflops", "GFLOPS", os.path.join(args.output_dir, "gflops.png"))
    plot_metric(df, "gbyte_s", "Effective GB/s", os.path.join(args.output_dir, "bandwidth.png"))

    if args.baseline in df["name"].unique():
        plot_speedup(df, args.baseline, os.path.join(args.output_dir, "speedup.png"))

    print(f"Plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
