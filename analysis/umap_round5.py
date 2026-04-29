import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

DATA_DIR = "/media/manukrishnan/Mk/prosperity_4/data/round5"

# Load prices
dfs = []
for day in [2, 3, 4]:
    df = pd.read_csv(f"{DATA_DIR}/prices_round_5_day_{day}.csv", sep=";")
    df["day"] = day
    dfs.append(df)
prices = pd.concat(dfs, ignore_index=True)

# Pivot mid_price: rows = (day, timestamp), cols = product
pivot = prices.pivot_table(index=["day", "timestamp"], columns="product", values="mid_price")
pivot = pivot.sort_index().ffill()

# Compute returns
returns = pivot.pct_change().fillna(0)

# Feature matrix: each product = its return series (transposed)
# shape: (n_products, n_timepoints)
X = returns.T.values  # (50, T)

# Normalise per product (unit variance)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.T).T  # still (50, T)

# --- UMAP ---
reducer = umap.UMAP(n_neighbors=5, min_dist=0.1, metric="correlation", random_state=42)
embedding = reducer.fit_transform(X_scaled)  # (50, 2)

products = returns.columns.tolist()

# Derive group labels: first word(s) that are shared across products
# All products follow FAMILY_VARIANT, where FAMILY is the prefix before the last _-separated variant
# But GALAXY_SOUNDS, OXYGEN_SHAKE, SLEEP_POD, UV_VISOR are 2-word prefixes
_two_word = {"GALAXY_SOUNDS", "OXYGEN_SHAKE", "SLEEP_POD", "UV_VISOR"}
def get_family(p):
    parts = p.split("_")
    if "_".join(parts[:2]) in _two_word:
        return "_".join(parts[:2])
    return parts[0]
groups = [get_family(p) for p in products]
unique_groups = sorted(set(groups))
group_to_int = {g: i for i, g in enumerate(unique_groups)}
colors = [group_to_int[g] for g in groups]

cmap = cm.get_cmap("tab10", len(unique_groups))

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# --- Plot 1: colour by product family ---
ax = axes[0]
for g in unique_groups:
    idx = [i for i, grp in enumerate(groups) if grp == g]
    ax.scatter(
        embedding[idx, 0], embedding[idx, 1],
        c=[cmap(group_to_int[g])], label=g, s=80, edgecolors="k", linewidths=0.4
    )
    for i in idx:
        ax.annotate(products[i].split("_")[-1], (embedding[i, 0], embedding[i, 1]),
                    fontsize=5.5, ha="center", va="bottom")
ax.set_title("UMAP — coloured by product family")
ax.legend(loc="best", fontsize=6, ncol=2)
ax.set_xlabel("UMAP-1")
ax.set_ylabel("UMAP-2")

# --- Plot 2: KMeans clusters ---
ax = axes[1]
n_clusters = len(unique_groups)
km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
km_labels = km.fit_predict(embedding)
scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=km_labels, cmap="tab10", s=80, edgecolors="k", linewidths=0.4)
for i, p in enumerate(products):
    ax.annotate(p, (embedding[i, 0], embedding[i, 1]), fontsize=4.5, ha="center", va="bottom")
ax.set_title(f"UMAP — KMeans (k={n_clusters})")
ax.set_xlabel("UMAP-1")
ax.set_ylabel("UMAP-2")
plt.colorbar(scatter, ax=ax, label="cluster")

plt.tight_layout()
out = "/media/manukrishnan/Mk/prosperity_4/analysis/umap_round5.png"
plt.savefig(out, dpi=150)
print(f"Saved: {out}")

# --- Print cluster membership ---
cluster_df = pd.DataFrame({"product": products, "family": groups, "km_cluster": km_labels})
cluster_df = cluster_df.sort_values(["km_cluster", "family"])
print("\nKMeans cluster membership:")
print(cluster_df.to_string(index=False))

# --- Correlation heatmap between families ---
family_returns = {}
for g in unique_groups:
    idx = [i for i, grp in enumerate(groups) if grp == g]
    family_returns[g] = X_scaled[idx].mean(axis=0)
fam_df = pd.DataFrame(family_returns)
corr = fam_df.corr()
print("\nFamily-level correlation matrix:")
print(corr.round(3).to_string())

# --- Intra vs inter family correlation ---
print("\nMean intra-family pairwise correlation:")
for g in unique_groups:
    idx = [i for i, grp in enumerate(groups) if grp == g]
    if len(idx) < 2:
        continue
    sub = X_scaled[idx]
    c = np.corrcoef(sub)
    np.fill_diagonal(c, np.nan)
    print(f"  {g:40s}: {np.nanmean(c):.4f}")
