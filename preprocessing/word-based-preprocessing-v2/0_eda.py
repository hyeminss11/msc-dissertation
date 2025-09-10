import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- Settings ----
INPUT_JSON = "preprocessing/word-based-preprocessing-v2/crossnews_gold_clean_anon.json"     # input file path
OUTPUT_TXT  = "preprocessing/word-based-preprocessing-v2/gold_final_stats_v2.txt"         # output text path
OUTPUT_HIST = "preprocessing/word-based-preprocessing-v2/gold_clean_wordcount_hist.png"          # output histogram path

# ---- JSON loader (supports array JSON and JSON Lines) ----
def load_json_to_df(path):
    try:
        # JSON Lines (each line is a JSON object)
        df = pd.read_json(path, lines=True)
    except ValueError:
        # Array JSON
        df = pd.read_json(path)
    # Ensure required columns
    for col in ['author', 'text', 'genre']:
        if col not in df.columns:
            df[col] = None
    return df[['author', 'text', 'genre']]

# ---- Compute descriptive statistics for word counts ----
def compute_word_stats(arr: np.ndarray):
    if arr.size == 0:
        return {
            'min': 0, 'max': 0, 'mean': 0.0,
            '25%': 0.0, '50%': 0.0, '75%': 0.0, '90%': 0.0
        }
    return {
        'min': int(np.min(arr)),
        'max': int(np.max(arr)),
        'mean': float(np.mean(arr)),
        '25%': float(np.percentile(arr, 25)),
        '50%': float(np.percentile(arr, 50)),
        '75%': float(np.percentile(arr, 75)),
        '90%': float(np.percentile(arr, 90)),
    }

# ---- Load data ----
df = load_json_to_df(INPUT_JSON)

# ---- Preprocessing ----
# Treat NaN or whitespace-only text as empty
text_series = df['text'].astype(str).fillna('').apply(lambda s: s.strip())
empty_text_count_all = (text_series == '').sum()

# Count words (whitespace-based split)
word_counts = text_series.apply(lambda s: 0 if s == '' else len(s.split()))
wc_np_all = word_counts.to_numpy()

# Unique authors
author_count_all = df['author'].nunique(dropna=True)

# Total number of texts
text_count_all = len(text_series)

# ---- Overall statistics ----
stats_all = compute_word_stats(wc_np_all)

# ---- Per-genre statistics ----
df['genre_filled'] = df['genre'].astype(str).fillna('UNKNOWN').replace({'None': 'UNKNOWN', 'nan': 'UNKNOWN'})
grouped = df.groupby('genre_filled', dropna=False)

genre_sections = []
for gname, gdf in grouped:
    gtext = gdf['text'].astype(str).fillna('').apply(lambda s: s.strip())
    g_empty = (gtext == '').sum()
    gwc = gtext.apply(lambda s: 0 if s == '' else len(s.split())).to_numpy()
    g_authors = gdf['author'].nunique(dropna=True)
    g_stats = compute_word_stats(gwc)

    lines = []
    lines.append(f"[Genre] {gname}")
    lines.append(f"Number of texts: {len(gtext)}")
    lines.append(f"Number of authors: {g_authors}")
    lines.append(f"Number of empty texts: {g_empty}")
    lines.append("Word Count Statistics:")
    lines.append(f"min: {g_stats['min']}")
    lines.append(f"max: {g_stats['max']}")
    lines.append(f"mean: {g_stats['mean']:.4f}")
    lines.append(f"25%: {g_stats['25%']:.4f}")
    lines.append(f"50% (median): {g_stats['50%']:.4f}")
    lines.append(f"75%: {g_stats['75%']:.4f}")
    lines.append(f"90%: {g_stats['90%']:.4f}")
    genre_sections.append("\n".join(lines))

# ---- Build result string ----
out = []
out.append("=== Overall ===")
out.append(f"Number of texts: {text_count_all}")
out.append(f"Number of authors: {author_count_all}")
out.append(f"Number of empty texts: {empty_text_count_all}")
out.append("Word Count Statistics:")
out.append(f"min: {stats_all['min']}")
out.append(f"max: {stats_all['max']}")
out.append(f"mean: {stats_all['mean']:.4f}")
out.append(f"25%: {stats_all['25%']:.4f}")
out.append(f"50% (median): {stats_all['50%']:.4f}")
out.append(f"75%: {stats_all['75%']:.4f}")
out.append(f"90%: {stats_all['90%']:.4f}")
out.append("")  # blank line
out.append("=== By Genre ===")
# Sort genres for consistent output
out.extend(["\n" + s for s in sorted(genre_sections, key=lambda x: x.splitlines()[0].lower())])

# ---- Save stats to TXT ----
with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(out))

print(f"Saved stats (overall + by genre) to {OUTPUT_TXT}")

# ===== Improved distribution visualizations =====
import numpy as np
import matplotlib.pyplot as plt

# Helper: Freedman–Diaconis rule for the number of bins
def fd_bins(x: np.ndarray):
    x = x[~np.isnan(x)]
    if x.size < 2:
        return 10
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        return 50
    bin_width = 2 * iqr / (x.size ** (1/3))
    if bin_width <= 0:
        return 50
    bins = int(np.ceil((x.max() - x.min()) / bin_width))
    return max(bins, 10)

# 0-length texts often dominate the leftmost bin; exclude them for shape analysis
wc_positive = wc_np_all[wc_np_all > 0]

# 1) Log-scaled x-axis histogram (excluding zeros)
plt.figure(figsize=(8, 5))
plt.hist(wc_positive, bins=fd_bins(wc_positive))
plt.xscale('log')
plt.title("Word Count Distribution (x-axis log, zeros excluded)")
plt.xlabel("Number of words per text (log scale)")
plt.ylabel("Frequency")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("preprocessing/word-based-preprocessing-v2/gold_wordcount_hist_logx.png", dpi=300)
plt.close()

# 2) Clipped histogram between p1 and p99 (linear x-axis)
p1, p99 = np.percentile(wc_positive, [1, 99])
wc_clip = wc_positive[(wc_positive >= p1) & (wc_positive <= p99)]
plt.figure(figsize=(8, 5))
plt.hist(wc_clip, bins=fd_bins(wc_clip))
plt.title(f"Word Count Distribution (clipped to 1–99th percentile: {int(p1)}–{int(p99)})")
plt.xlabel("Number of words per text")
plt.ylabel("Frequency")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("preprocessing/word-based-preprocessing-v2/gold_wordcount_hist_clip_p1p99.png", dpi=300)
plt.close()

# 3) ECDF (empirical cumulative distribution function)
def ecdf(x: np.ndarray):
    x = np.sort(x)
    y = np.arange(1, x.size + 1) / x.size
    return x, y

x_ecdf, y_ecdf = ecdf(wc_positive)
plt.figure(figsize=(8, 5))
plt.plot(x_ecdf, y_ecdf, drawstyle="steps-post")
plt.title("ECDF of Word Counts (zeros excluded)")
plt.xlabel("Number of words per text")
plt.ylabel("Cumulative probability")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("preprocessing/word-based-preprocessing-v2/gold_wordcount_ecdf.png", dpi=300)
plt.close()

# 4) Per-genre boxplots on log scale (exclude zeros)
#    Build a list in the same (sorted) order used above for text output
genres_sorted = sorted(grouped.groups.keys(), key=lambda x: str(x).lower())
data_per_genre = []
labels_per_genre = []
for g in genres_sorted:
    arr = df.loc[df['genre_filled'] == g, 'text'].astype(str).str.strip().apply(
        lambda s: 0 if s == '' else len(s.split())
    ).to_numpy()
    arr = arr[arr > 0]
    if arr.size > 0:
        data_per_genre.append(arr)
        labels_per_genre.append(str(g))

if len(data_per_genre) > 0:
    plt.figure(figsize=(10, 5))
    bp = plt.boxplot(data_per_genre, showfliers=False, labels=labels_per_genre, vert=True)
    plt.yscale('log')
    plt.title("Word Counts by Genre (boxplot, log y, zeros excluded)")
    plt.xlabel("Genre")
    plt.ylabel("Number of words per text (log scale)")
    plt.grid(True, axis='y', linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("preprocessing/word-based-preprocessing-v2/gold_wordcount_boxplot_by_genre_logy.png", dpi=300)
    plt.close()