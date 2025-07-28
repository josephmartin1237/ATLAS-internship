import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit


sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

ETmiss_min = float(input("Enter minimum ETmiss cut [GeV]: "))
ETmiss_max = float(input("Enter maximum ETmiss cut [GeV]: "))
N_bjets_cut = int(input("Enter maximum number of b-jets allowed: "))

CUTS = {
    "ETmiss_min": ETmiss_min,
    "ETmiss_max": ETmiss_max,
    "N_bjets": N_bjets_cut
}

SIGNAL_PREFIX = "DM_"
EXCLUDED_SIGNALS = ["DM_10"]
BACKGROUND_FILES = ["ZZ.csv", "WZ.csv", "Z+jets.csv", "Non-resonant_ll.csv"]
VARIABLES_TO_PLOT = ["ETmiss", "mll", "lead_lep_pt", "dRll"]
VARIABLE_LABELS = {
    "ETmiss": "Missing Transverse Energy $E_T^{miss}$ [GeV]",
    "mll": "Invariant Mass of Lepton Pair $m_{\\ell\\ell}$ [GeV]",
    "lead_lep_pt": "Leading Lepton $p_T$ [GeV]",
    "dRll": "$\\Delta R_{\\ell\\ell}$ (Lepton Angular Separation)"
}

MASS_ORDER = [100, 200, 300, 400, 500, 600, 700, 800, 2000]


def crystal_ball(x, alpha, n, mean, sigma, A):
    z = (x - mean) / sigma
    abs_alpha = abs(alpha)
    A1 = (n / abs_alpha)**n * np.exp(-abs_alpha**2 / 2)
    B1 = n / abs_alpha - abs_alpha

    result = np.where(
        z > -alpha,
        A * np.exp(-0.5 * z**2),
        A * A1 * (B1 - z)**(-n)
    )
    return result


def fit_crystal_ball(df, variable, weight_column="totalWeight", title=""):
    df_cut = apply_cuts(df)
    if df_cut.empty:
        print("No data after cuts for fitting.")
        return

    data = df_cut[variable]
    weights = df_cut[weight_column]

    bins = 50
    counts, bin_edges = np.histogram(data, bins=bins, weights=weights, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    A_init = max(counts)
    mean_init = np.average(data, weights=weights)
    sigma_init = np.sqrt(np.average((data - mean_init)**2, weights=weights))
    alpha_init = 1.5
    n_init = 3

    p0 = [alpha_init, n_init, mean_init, sigma_init, A_init]

    try:
        popt, pcov = curve_fit(crystal_ball, bin_centers, counts, p0=p0)
        perr = np.sqrt(np.diag(pcov))
    except RuntimeError:
        print("Fit did not converge.")
        return

    plt.figure(figsize=(8, 5))
    plt.bar(bin_centers, counts, width=(bin_edges[1]-bin_edges[0]), alpha=0.6, label="Data (weighted)")

    x_fit = np.linspace(bin_edges[0], bin_edges[-1], 500)
    y_fit = crystal_ball(x_fit, *popt)

    plt.plot(x_fit, y_fit, 'r-', label="Crystal Ball Fit")
    plt.title(title or f"Crystal Ball fit to {variable}")
    plt.xlabel(VARIABLE_LABELS.get(variable, variable))
    plt.ylabel("Weighted counts")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Fit parameters:\n"
          f" alpha = {popt[0]:.3f} ± {perr[0]:.3f}\n"
          f" n     = {popt[1]:.3f} ± {perr[1]:.3f}\n"
          f" mean  = {popt[2]:.3f} ± {perr[2]:.3f}\n"
          f" sigma = {popt[3]:.3f} ± {perr[3]:.3f}\n"
          f" A     = {popt[4]:.1f} ± {perr[4]:.1f}")


def load_files():
    files = os.listdir()
    signal_data = {}
    background_data = {}
    for f in files:
        if f.endswith(".csv"):
            if f.startswith(SIGNAL_PREFIX):
                label = f.replace(".csv", "")
                if label not in EXCLUDED_SIGNALS:
                    signal_data[label] = pd.read_csv(f)
            elif f in BACKGROUND_FILES:
                label = f.replace(".csv", "")
                background_data[label] = pd.read_csv(f)
    return signal_data, background_data

def apply_cuts(df, etmin=None, etmax=None, nbjet=None):
    etmin = etmin if etmin is not None else CUTS["ETmiss_min"]
    etmax = etmax if etmax is not None else CUTS["ETmiss_max"]
    nbjet = nbjet if nbjet is not None else CUTS["N_bjets"]
    return df[(df["ETmiss"] >= etmin) & (df["ETmiss"] <= etmax) & (df["N_bjets"] <= nbjet)]

def calculate_significance(s, b):
    if s + b > 0:
        return s / np.sqrt(s + b)
    return 0.0

def calculate_significance_uncertainty(s, b):
    if s > 0 and b > 0:
        denom = (s + b)**1.5
        dZ_dS = (s + b/2) / denom
        dZ_dB = -s / (2 * denom)
        delta_S = np.sqrt(s)
        delta_B = np.sqrt(b)
        return np.sqrt((dZ_dS * delta_S)**2 + (dZ_dB * delta_B)**2)
    return 0.0

def extract_mass(label):
    try:
        return int(label.split('_')[1])
    except:
        return 9999

def mass_sort_key(label):
    mass = extract_mass(label)
    return MASS_ORDER.index(mass) if mass in MASS_ORDER else 9999

def plot_variable(signal_data, background_data, variable, title):
    plt.figure()
    for bkg_label, df in background_data.items():
        df_cut = apply_cuts(df)
        label_map = {
            "ZZ": "ZZ → ℓℓνν",
            "WZ": "WZ → ℓνℓℓ",
            "Z+jets": "Z + jets",
            "Non-resonant_ll": "Non-resonant ℓℓ"
        }
        pretty_label = label_map.get(bkg_label, bkg_label)
        sns.histplot(
            data=df_cut,
            x=variable,
            weights=df_cut["totalWeight"],
            label=pretty_label,
            stat="count",
            bins=100,
            kde=False,
            alpha=0.5,
            element="step",
            fill=True
        )
    for label, signal_df in signal_data.items():
        df_cut = apply_cuts(signal_df)
        sns.histplot(
            data=df_cut,
            x=variable,
            weights=df_cut["totalWeight"],
            label=label,
            stat="count",
            bins=100,
            kde=False,
            alpha=0.7,
            element="step"
        )
    plt.title(title)
    plt.xlabel(VARIABLE_LABELS.get(variable, variable))
    plt.ylabel("Weighted Events")
    plt.legend(title="Process")
    plt.tight_layout()
    plt.show()

def run_cut_scan(signal_data, background_data):
    etmiss_ranges = list(range(100, 501, 25))
    bjet_cuts = [0, 1, 2]
    all_results = []

    sorted_signal_items = sorted(signal_data.items(), key=lambda x: mass_sort_key(x[0]))
    for label, signal_df in sorted_signal_items:
        results = []
        for etmin in etmiss_ranges:
            for bcut in bjet_cuts:
                s = apply_cuts(signal_df, etmin, None, bcut)["totalWeight"].sum()
                b = sum(apply_cuts(df, etmin, None, bcut)["totalWeight"].sum() for df in background_data.values())
                sig = calculate_significance(s, b)
                sig_unc = calculate_significance_uncertainty(s, b)
                sb_ratio = s / b if b > 0 else 0
                results.append({"ETmiss_cut": etmin, "N_bjets_cut": bcut, "Significance": sig, "Uncertainty": sig_unc, "S_over_B": sb_ratio})
                all_results.append((label, etmin, bcut, sig, sig_unc))

        df = pd.DataFrame(results)
        heatmap_df = df.pivot(index="N_bjets_cut", columns="ETmiss_cut", values="Significance")
        plt.figure(figsize=(8, 5))
        sns.heatmap(heatmap_df, annot=True, cmap="viridis", cbar_kws={"label": "Significance σ"}, fmt=".2f")
        plt.title(f"Significance Heatmap for {label}")
        plt.xlabel("ETmiss Cut [GeV]")
        plt.ylabel("Max N_bjets")
        plt.tight_layout()
        plt.show()

    plt.figure(figsize=(10, 6))
    all_df = pd.DataFrame(all_results, columns=["Signal", "ETmiss_cut", "N_bjets_cut", "Significance", "Uncertainty"])
    all_df = all_df[all_df["N_bjets_cut"] == 0]
    sorted_signals = sorted(all_df["Signal"].unique(), key=mass_sort_key)
    for signal in sorted_signals:
        sub = all_df[all_df["Signal"] == signal].sort_values("ETmiss_cut")
        plt.errorbar(sub["ETmiss_cut"], sub["Significance"], yerr=sub["Uncertainty"],
                     label=signal, marker='o', capsize=3)
    plt.title("Significance vs ETmiss Cut (0 bjets only) with Uncertainties")
    plt.xlabel("ETmiss Cut [GeV]")
    plt.ylabel("Significance σ")
    plt.ylim(0, 7)
    plt.legend(title="Signal Mass")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    grouped = all_df.loc[all_df.groupby("Signal")["Significance"].idxmax()]
    grouped = grouped.loc[sorted(grouped.index, key=lambda i: mass_sort_key(grouped.loc[i, "Signal"]))]
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(grouped["Signal"], grouped["Significance"], c=grouped["Significance"], cmap="viridis", s=200, edgecolor='black')
    plt.errorbar(grouped["Signal"], grouped["Significance"], yerr=grouped["Uncertainty"], fmt='none', ecolor='gray', alpha=0.7)
    plt.xticks(rotation=45)
    plt.colorbar(scatter, label="Max Significance σ")
    plt.title("Maximum Significance per DM Mass Sample (0 bjets only)")
    plt.xlabel("Signal Sample")
    plt.ylabel("Significance σ")
    plt.ylim(0, 7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    signal_data, background_data = load_files()

    for var in VARIABLES_TO_PLOT:
        plot_variable(signal_data, background_data, var, f"{VARIABLE_LABELS.get(var, var)}")

    run_cut_scan(signal_data, background_data)

    print("\n--- Crystal Ball fits for signal samples ---")
    for label, df in signal_data.items():
        print(f"\nFitting Crystal Ball to {label} sample:")
        fit_crystal_ball(df, "mll", title=f"Signal: {label} - Crystal Ball fit to $m_{{ll}}$")

    print("\n--- Crystal Ball fits for background samples ---")
    for label, df in background_data.items():
        print(f"\nFitting Crystal Ball to background {label}:")
        fit_crystal_ball(df, "mll", title=f"Background: {label} - Crystal Ball fit to $m_{{ll}}$")













    

