import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.interpolate import make_interp_spline
from sklearn.metrics import roc_curve, auc


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
    "mll": "Invariant Mass of Lepton Pair $m_{\ell\ell}$ [GeV]",
    "lead_lep_pt": "Leading Lepton $p_T$ [GeV]",
    "dRll": "$\\Delta R_{\ell\ell}$ (Lepton Angular Separation)"
}


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

def calculate_significance(signal_df, background_dict, etmin, bcut):
    s = apply_cuts(signal_df, etmin, None, bcut)["totalWeight"].sum()
    b = sum(apply_cuts(df, etmin, None, bcut)["totalWeight"].sum() for df in background_dict.values())
    if b > 0:
        sig = np.sqrt(2 * ((s + b) * np.log(1 + s / b) - s))
    else:
        sig = 0.0
    return min(sig, 3.5)

def plot_variable(signal_data, background_data, variable, title):
    plt.figure()
    total_weights = {}

    for bkg_label, df in background_data.items():
        df_cut = apply_cuts(df)
        label_map = {
            "ZZ": "ZZ → ℓℓνν",
            "WZ": "WZ → ℓνℓℓ",
            "Z+jets": "Z + jets",
            "Non-resonant_ll": "Non-resonant ℓℓ"
        }
        pretty_label = label_map.get(bkg_label, bkg_label)
        weight_sum = df_cut["totalWeight"].sum()
        total_weights[pretty_label] = weight_sum
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
        weight_sum = df_cut["totalWeight"].sum()
        total_weights[label] = weight_sum
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

def generate_yield_table(data_dict, label, background_data=None):
    for name, signal_df in data_dict.items():
        signal_sum = apply_cuts(signal_df)["totalWeight"].sum()
        bkg_sum = sum(apply_cuts(df)["totalWeight"].sum() for df in background_data.values()) if background_data else 0

        print(f"\n=== Event Yields: {name} ===")
        print("{:<25} {:>20} {:>20}".format("Process (Event Type)", "Before Cuts", "After Cuts"))
        print("{:<25} {:>20} {:>20}".format(name, round(signal_df["totalWeight"].sum(), 2), round(signal_sum, 2)))

        if background_data:
            for bkg_name, df in background_data.items():
                after_cuts = apply_cuts(df)["totalWeight"].sum()
                label_map = {
                    "ZZ": "ZZ → ℓℓνν",
                    "WZ": "WZ → ℓνℓℓ",
                    "Z+jets": "Z + jets",
                    "Non-resonant_ll": "Non-resonant ℓℓ",
                }
                event_type = label_map.get(bkg_name, bkg_name)
                print("{:<25} {:>20} {:>20}".format(event_type, round(df["totalWeight"].sum(), 2), round(after_cuts, 2)))

        if signal_sum > 0 and bkg_sum > 0:
            significance = np.sqrt(2 * ((signal_sum + bkg_sum) * np.log(1 + signal_sum / bkg_sum) - signal_sum))
            print(f"\nEstimated Significance: {min(significance, 3.5):.2f}σ\n")

def run_cut_scan(signal_data, background_data):
    etmiss_ranges = list(range(100, 501, 25))
    bjet_cuts = [0, 1, 2]
    all_results = []

    for label, signal_df in signal_data.items():
        results = []
        for etmin in etmiss_ranges:
            for bcut in bjet_cuts:
                sig = calculate_significance(signal_df, background_data, etmin, bcut)
                results.append((etmin, bcut, sig))
                all_results.append((label, etmin, bcut, sig))

        df = pd.DataFrame(results, columns=["ETmiss_cut", "N_bjets_cut", "Significance"])

        best_row = df.loc[df["Significance"].idxmax()]
        print(f"\nBest cut for {label}: ETmiss > {best_row['ETmiss_cut']} GeV, N_bjets <= {int(best_row['N_bjets_cut'])}, Significance = {best_row['Significance']:.2f}σ")

        # Scatter plot
        plt.figure(figsize=(8, 5))
        scatter = plt.scatter(df["ETmiss_cut"], df["N_bjets_cut"], c=df["Significance"], cmap="viridis", s=150, edgecolor='k')
        plt.scatter(best_row["ETmiss_cut"], best_row["N_bjets_cut"], color="red", s=200, label="Best Cut", marker="X")
        plt.colorbar(scatter, label="Significance σ")
        plt.title(f"Significance Scatter for {label}")
        plt.xlabel("ETmiss Cut [GeV]")
        plt.ylabel("Max N_bjets")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Heatmap
        heatmap_df = df.pivot(index="N_bjets_cut", columns="ETmiss_cut", values="Significance")
        plt.figure(figsize=(8, 5))
        sns.heatmap(heatmap_df, annot=True, cmap="viridis", cbar_kws={"label": "Significance σ"}, fmt=".2f")
        plt.title(f"Significance Heatmap for {label}")
        plt.xlabel("ETmiss Cut [GeV]")
        plt.ylabel("Max N_bjets")
        plt.tight_layout()
        plt.show()

    # Unified scatter 
    all_df = pd.DataFrame(all_results, columns=["Signal", "ETmiss_cut", "N_bjets_cut", "Significance"])
    plt.figure(figsize=(10, 6))
    for signal in all_df["Signal"].unique():
        for ncut in sorted(all_df["N_bjets_cut"].unique()):
            sub = all_df[(all_df["Signal"] == signal) & (all_df["N_bjets_cut"] == ncut)]
            if len(sub) >= 4:
                x_sorted = np.array(sorted(sub["ETmiss_cut"]))
                y_sorted = np.array(sub.sort_values("ETmiss_cut")["Significance"])
                spline = make_interp_spline(x_sorted, y_sorted)
                x_new = np.linspace(min(x_sorted), max(x_sorted), 300)
                y_smooth = spline(x_new)
                plt.plot(x_new, y_smooth, label=f"{signal} (N_bjets ≤ {ncut})")
            else:
                plt.plot(sub["ETmiss_cut"], sub["Significance"], label=f"{signal} (N_bjets ≤ {ncut})")

    plt.title("Comparison of Significance Across Signal Masses")
    plt.xlabel("ETmiss Cut [GeV]")
    plt.ylabel("Significance σ")
    plt.legend(title="Signal Mass")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    signal_data, background_data = load_files()

    for var in VARIABLES_TO_PLOT:
        plot_variable(signal_data, background_data, var, f"{VARIABLE_LABELS.get(var, var)}")

    generate_yield_table(signal_data, "Signal", background_data)

    run_cut_scan(signal_data, background_data)









