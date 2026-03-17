import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .config import (
    BASELINE_TYPE1_PATH,
    BASELINE_TYPE2_PATH,
    OUTPUTS_TYPE1_PATH,
    OUTPUTS_TYPE2_PATH,
    REFERENCE_TYPE1_PATH,
    REFERENCE_TYPE2_PATH,
    METRICS_SUMMARY_CSV,
    METRICS_STRUCTURE_CSV,
    METRICS_CROSS_STYLE_CSV,
    PLOT_COSINE_PATH,
    PLOT_STYLE_SCORE_PATH,
    PLOT_MARGIN_PATH,
    PLOT_CROSS_STYLE_PATH,
    RESULTS_DRAFT_PATH,
    MODEL_NAME,
    TRAIN_SIZE,
    NUM_EPOCHS,
)
from .io_utils import load_lines

BG_FIG = "#0b1220"
BG_AX = "#111827"
GRID = "#334155"
TEXT = "#e5e7eb"

COLOR_BEFORE = "#94a3b8"
COLOR_AFTER = "#22d3ee"
HEATMAP_CMAP = "mako"


def apply_dark_report_style():
    sns.set_theme(style="darkgrid", context="talk")
    plt.rcParams.update({
        "figure.facecolor": BG_FIG,
        "axes.facecolor": BG_AX,
        "savefig.facecolor": BG_FIG,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "text.color": TEXT,
        "axes.titlecolor": TEXT,
        "grid.color": GRID,
        "grid.alpha": 0.35,
        "axes.titleweight": "bold",
        "legend.frameon": True,
        "legend.facecolor": BG_AX,
        "legend.edgecolor": GRID,
    })


def count_sentences(text):
    if not text.strip():
        return 0
    return max(1, len(re.findall(r"[.!?]+", text)))


def avg_words(texts):
    return float(np.mean([len(t.split()) for t in texts])) if texts else 0.0


def avg_chars(texts):
    return float(np.mean([len(t) for t in texts])) if texts else 0.0


def avg_sentences(texts):
    return float(np.mean([count_sentences(t) for t in texts])) if texts else 0.0


EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U000024C2-\U0001F251"
    "\U0001F1E6-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)


def has_emoji(text):
    return bool(EMOJI_RE.search(text or ""))


def starts_with_emoji(text):
    text = (text or "").strip()
    if not text:
        return False
    return has_emoji(text[:4]) or has_emoji(text[:6])


def rate_banki_tag(texts):
    if not texts:
        return 0.0
    return float(np.mean(["@banki_oil" in t.lower() for t in texts]))


def rate_no_emoji(texts):
    if not texts:
        return 0.0
    return float(np.mean([not has_emoji(t) for t in texts]))


def rate_leading_emoji(texts):
    if not texts:
        return 0.0
    return float(np.mean([starts_with_emoji(t) for t in texts]))


def mean_pairwise_cosine(emb_a, emb_b):
    sims = []
    for a, b in zip(emb_a, emb_b):
        sims.append(cosine_similarity([a], [b])[0][0])
    return float(np.mean(sims))


def mean_embedding(embs):
    return np.mean(embs, axis=0, keepdims=True)


def annotate_bars(ax, fmt="{:.3f}", suffix="", y_offset=4):
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            fmt.format(height) + suffix,
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=10,
            color=TEXT,
            xytext=(0, y_offset),
            textcoords="offset points",
        )


def grouped_barplot(plot_df, title, ylabel, save_path, ylim=None):
    apply_dark_report_style()

    plt.figure(figsize=(9, 5))
    ax = sns.barplot(
        data=plot_df,
        x="channel",
        y="score",
        hue="stage",
        palette=[COLOR_BEFORE, COLOR_AFTER],
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")

    if ylim is not None:
        ax.set_ylim(*ylim)

    legend = ax.legend(title="")
    if legend is not None:
        legend.get_frame().set_facecolor(BG_AX)
        legend.get_frame().set_edgecolor(GRID)
        for text in legend.get_texts():
            text.set_color(TEXT)

    annotate_bars(ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def heatmap_plot(cross_df, save_path):
    apply_dark_report_style()

    plt.figure(figsize=(6.8, 5.2))
    ax = sns.heatmap(
        cross_df,
        annot=True,
        fmt=".3f",
        cmap=HEATMAP_CMAP,
        vmin=min(0.25, float(cross_df.values.min())),
        vmax=1.0,
        linewidths=1.0,
        linecolor=GRID,
        annot_kws={"color": TEXT, "fontsize": 12, "fontweight": "bold"},
    )

    ax.set_title("Cross-Style Similarity Matrix\nDiagonal should be higher", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(GRID)

    cbar = ax.collections[0].colorbar
    cbar.outline.set_edgecolor(GRID)
    cbar.ax.yaxis.set_tick_params(color=TEXT)
    plt.setp(cbar.ax.get_yticklabels(), color=TEXT)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def compute_metrics(train_banki=None, train_moscow=None):
    ref_type1 = load_lines(REFERENCE_TYPE1_PATH)
    ref_type2 = load_lines(REFERENCE_TYPE2_PATH)

    base_type1 = load_lines(BASELINE_TYPE1_PATH)
    base_type2 = load_lines(BASELINE_TYPE2_PATH)
    out_type1 = load_lines(OUTPUTS_TYPE1_PATH)
    out_type2 = load_lines(OUTPUTS_TYPE2_PATH)

    print("""
METRICS EXPLANATION
===================

1) Cosine similarity to held-out reference
   - Each generated post is compared with the real held-out post.
   - HIGHER = BETTER.

2) Cross-style similarity matrix
   - Generated Type1 should be closer to Reference Type1 than to Reference Type2.
   - Generated Type2 should be closer to Reference Type2 than to Reference Type1.
   - DIAGONAL should be higher.

3) Style gap before/after
   - Own-style similarity minus other-style similarity.
   - HIGHER = BETTER.

4) Marker compliance
   - Type1: @banki_oil rate and no-emoji rate.
   - Type2: leading-emoji rate.
   - HIGHER = BETTER.

5) Structure closeness to references
   - How close outputs are to reference average length and sentence count.
   - LOWER distance = BETTER.
""")

    embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    emb_ref1 = embedder.encode(ref_type1, show_progress_bar=False, normalize_embeddings=True)
    emb_ref2 = embedder.encode(ref_type2, show_progress_bar=False, normalize_embeddings=True)
    emb_base1 = embedder.encode(base_type1, show_progress_bar=False, normalize_embeddings=True)
    emb_base2 = embedder.encode(base_type2, show_progress_bar=False, normalize_embeddings=True)
    emb_out1 = embedder.encode(out_type1, show_progress_bar=False, normalize_embeddings=True)
    emb_out2 = embedder.encode(out_type2, show_progress_bar=False, normalize_embeddings=True)

    cos_base1 = mean_pairwise_cosine(emb_base1, emb_ref1)
    cos_out1 = mean_pairwise_cosine(emb_out1, emb_ref1)

    cos_base2 = mean_pairwise_cosine(emb_base2, emb_ref2)
    cos_out2 = mean_pairwise_cosine(emb_out2, emb_ref2)

    cross_base_11 = cosine_similarity(mean_embedding(emb_base1), mean_embedding(emb_ref1))[0][0]
    cross_base_12 = cosine_similarity(mean_embedding(emb_base1), mean_embedding(emb_ref2))[0][0]
    cross_base_21 = cosine_similarity(mean_embedding(emb_base2), mean_embedding(emb_ref1))[0][0]
    cross_base_22 = cosine_similarity(mean_embedding(emb_base2), mean_embedding(emb_ref2))[0][0]

    cross_11 = cosine_similarity(mean_embedding(emb_out1), mean_embedding(emb_ref1))[0][0]
    cross_12 = cosine_similarity(mean_embedding(emb_out1), mean_embedding(emb_ref2))[0][0]
    cross_21 = cosine_similarity(mean_embedding(emb_out2), mean_embedding(emb_ref1))[0][0]
    cross_22 = cosine_similarity(mean_embedding(emb_out2), mean_embedding(emb_ref2))[0][0]

    gap_before_type1 = cross_base_11 - cross_base_12
    gap_after_type1 = cross_11 - cross_12

    gap_before_type2 = cross_base_22 - cross_base_21
    gap_after_type2 = cross_22 - cross_21

    cross_df = pd.DataFrame(
        [[cross_11, cross_12], [cross_21, cross_22]],
        index=["Generated Type1", "Generated Type2"],
        columns=["Reference Type1", "Reference Type2"],
    )

    banki_tag_before = rate_banki_tag(base_type1)
    banki_tag_after = rate_banki_tag(out_type1)

    banki_no_emoji_before = rate_no_emoji(base_type1)
    banki_no_emoji_after = rate_no_emoji(out_type1)

    moscow_emoji_before = rate_leading_emoji(base_type2)
    moscow_emoji_after = rate_leading_emoji(out_type2)

    ref1_words = avg_words(ref_type1)
    ref2_words = avg_words(ref_type2)

    ref1_sents = avg_sentences(ref_type1)
    ref2_sents = avg_sentences(ref_type2)

    base1_words = avg_words(base_type1)
    out1_words = avg_words(out_type1)
    base2_words = avg_words(base_type2)
    out2_words = avg_words(out_type2)

    base1_sents = avg_sentences(base_type1)
    out1_sents = avg_sentences(out_type1)
    base2_sents = avg_sentences(base_type2)
    out2_sents = avg_sentences(out_type2)

    word_dist_before_type1 = abs(base1_words - ref1_words)
    word_dist_after_type1 = abs(out1_words - ref1_words)
    word_dist_before_type2 = abs(base2_words - ref2_words)
    word_dist_after_type2 = abs(out2_words - ref2_words)

    sent_dist_before_type1 = abs(base1_sents - ref1_sents)
    sent_dist_after_type1 = abs(out1_sents - ref1_sents)
    sent_dist_before_type2 = abs(base2_sents - ref2_sents)
    sent_dist_after_type2 = abs(out2_sents - ref2_sents)

    structure_df = pd.DataFrame({
        "set": [
            "ref_type1", "baseline_type1", "styled_type1",
            "ref_type2", "baseline_type2", "styled_type2",
        ],
        "avg_words": [
            ref1_words, base1_words, out1_words,
            ref2_words, base2_words, out2_words,
        ],
        "avg_chars": [
            avg_chars(ref_type1), avg_chars(base_type1), avg_chars(out_type1),
            avg_chars(ref_type2), avg_chars(base_type2), avg_chars(out_type2),
        ],
        "avg_sentences": [
            ref1_sents, base1_sents, out1_sents,
            ref2_sents, base2_sents, out2_sents,
        ],
    })

    summary_df = pd.DataFrame({
        "channel": ["banki_oil", "moscowach"],

        "cosine_before": [cos_base1, cos_base2],
        "cosine_after": [cos_out1, cos_out2],
        "cosine_gain": [cos_out1 - cos_base1, cos_out2 - cos_base2],

        "style_gap_before": [gap_before_type1, gap_before_type2],
        "style_gap_after": [gap_after_type1, gap_after_type2],
        "style_gap_gain": [gap_after_type1 - gap_before_type1, gap_after_type2 - gap_before_type2],

        "final_to_own_ref": [cross_11, cross_22],
        "final_to_other_ref": [cross_12, cross_21],

        "main_marker_before": [banki_tag_before, moscow_emoji_before],
        "main_marker_after": [banki_tag_after, moscow_emoji_after],

        "aux_marker_before": [banki_no_emoji_before, np.nan],
        "aux_marker_after": [banki_no_emoji_after, np.nan],

        "word_dist_before": [word_dist_before_type1, word_dist_before_type2],
        "word_dist_after": [word_dist_after_type1, word_dist_after_type2],

        "sent_dist_before": [sent_dist_before_type1, sent_dist_before_type2],
        "sent_dist_after": [sent_dist_after_type1, sent_dist_after_type2],
    })

    print("CROSS-STYLE MATRIX")
    print("==================")
    print(cross_df.round(4).to_string())
    print()

    print("STRUCTURAL SANITY METRICS")
    print("=========================")
    print(structure_df.round(3).to_string(index=False))
    print()

    print("SUMMARY TABLE")
    print("=============")
    print(summary_df.round(4).to_string(index=False))

    summary_df.to_csv(METRICS_SUMMARY_CSV, index=False)
    structure_df.to_csv(METRICS_STRUCTURE_CSV, index=False)
    cross_df.to_csv(METRICS_CROSS_STYLE_CSV)

    plot_df_cos = pd.DataFrame({
        "channel": ["banki_oil", "banki_oil", "moscowach", "moscowach"],
        "stage": ["Before LoRA", "After LoRA", "Before LoRA", "After LoRA"],
        "score": [cos_base1, cos_out1, cos_base2, cos_out2],
    })
    grouped_barplot(
        plot_df_cos,
        "Cosine Similarity to Real Held-Out Posts\nHigher = Better",
        "Cosine similarity",
        PLOT_COSINE_PATH,
        ylim=(0.45, 1.0),
    )

    plot_df_gap = pd.DataFrame({
        "channel": ["banki_oil", "banki_oil", "moscowach", "moscowach"],
        "stage": ["Before LoRA", "After LoRA", "Before LoRA", "After LoRA"],
        "score": [gap_before_type1, gap_after_type1, gap_before_type2, gap_after_type2],
    })
    gap_min = float(min(plot_df_gap["score"])) - 0.05
    gap_max = float(max(plot_df_gap["score"])) + 0.05
    grouped_barplot(
        plot_df_gap,
        "Style Gap: Own Reference minus Other Reference\nHigher = Better",
        "Style gap",
        PLOT_MARGIN_PATH,
        ylim=(gap_min, gap_max),
    )

    marker_plot_df = pd.DataFrame({
        "channel": [
            "banki_oil (@banki_oil)",
            "banki_oil (@banki_oil)",
            "moscowach (leading emoji)",
            "moscowach (leading emoji)",
        ],
        "stage": ["Before LoRA", "After LoRA", "Before LoRA", "After LoRA"],
        "score": [banki_tag_before, banki_tag_after, moscow_emoji_before, moscow_emoji_after],
    })
    grouped_barplot(
        marker_plot_df,
        "Simple Marker Compliance\nHigher = Better",
        "Rate",
        PLOT_STYLE_SCORE_PATH,
        ylim=(0.0, 1.0),
    )

    heatmap_plot(cross_df, PLOT_CROSS_STYLE_PATH)

    return {
        "summary_df": summary_df,
        "structure_df": structure_df,
        "cross_df": cross_df,
        "examples": {
            "ref_type1": ref_type1,
            "ref_type2": ref_type2,
            "base_type1": base_type1,
            "base_type2": base_type2,
            "out_type1": out_type1,
            "out_type2": out_type2,
        },
    }


def print_interpretation(results):
    summary_df = results["summary_df"]
    cross_df = results["cross_df"]

    cos_base1 = summary_df.loc[0, "cosine_before"]
    cos_out1 = summary_df.loc[0, "cosine_after"]
    cos_base2 = summary_df.loc[1, "cosine_before"]
    cos_out2 = summary_df.loc[1, "cosine_after"]

    gap_before1 = summary_df.loc[0, "style_gap_before"]
    gap_after1 = summary_df.loc[0, "style_gap_after"]
    gap_before2 = summary_df.loc[1, "style_gap_before"]
    gap_after2 = summary_df.loc[1, "style_gap_after"]

    cross_11 = cross_df.iloc[0, 0]
    cross_12 = cross_df.iloc[0, 1]
    cross_21 = cross_df.iloc[1, 0]
    cross_22 = cross_df.iloc[1, 1]

    print("\nINTERPRETATION")
    print("==============")

    if cos_out1 > cos_base1:
        print(f"Type 1 cosine improved: {cos_base1:.4f} -> {cos_out1:.4f}")
    else:
        print(f"Type 1 cosine did NOT improve: {cos_base1:.4f} -> {cos_out1:.4f}")

    if cos_out2 > cos_base2:
        print(f"Type 2 cosine improved: {cos_base2:.4f} -> {cos_out2:.4f}")
    else:
        print(f"Type 2 cosine did NOT improve: {cos_base2:.4f} -> {cos_out2:.4f}")

    if gap_after1 > gap_before1:
        print(f"Type 1 style gap improved: {gap_before1:.4f} -> {gap_after1:.4f}")
    else:
        print(f"Type 1 style gap did NOT improve: {gap_before1:.4f} -> {gap_after1:.4f}")

    if gap_after2 > gap_before2:
        print(f"Type 2 style gap improved: {gap_before2:.4f} -> {gap_after2:.4f}")
    else:
        print(f"Type 2 style gap did NOT improve: {gap_before2:.4f} -> {gap_after2:.4f}")

    print("\nCross-style check:")
    print(f"Generated Type1 -> Ref Type1: {cross_11:.4f}")
    print(f"Generated Type1 -> Ref Type2: {cross_12:.4f}")
    print(f"Generated Type2 -> Ref Type1: {cross_21:.4f}")
    print(f"Generated Type2 -> Ref Type2: {cross_22:.4f}")

    if cross_11 > cross_12 and cross_22 > cross_21:
        print("PASS: each generated type is closer to its own reference type.")
    else:
        print("WARNING: cross-style separation is weak.")


def print_examples(test_type1, test_type2, results, limit=3):
    ref_type1 = results["examples"]["ref_type1"]
    ref_type2 = results["examples"]["ref_type2"]
    base_type1 = results["examples"]["base_type1"]
    base_type2 = results["examples"]["base_type2"]
    out_type1 = results["examples"]["out_type1"]
    out_type2 = results["examples"]["out_type2"]

    print("\n" + "=" * 90)
    print("EXAMPLES")
    print("=" * 90)

    for i in range(min(limit, len(test_type1))):
        print(f"\n--- TYPE 1 / Example {i+1} ---")
        print("INPUT (neutral):")
        print(test_type1[i]["input"])
        print("\nREFERENCE (real channel post):")
        print(ref_type1[i])
        print("\nBEFORE LoRA (baseline):")
        print(base_type1[i])
        print("\nAFTER LoRA (styled):")
        print(out_type1[i])

    for i in range(min(limit, len(test_type2))):
        print(f"\n--- TYPE 2 / Example {i+1} ---")
        print("INPUT (neutral):")
        print(test_type2[i]["input"])
        print("\nREFERENCE (real channel post):")
        print(ref_type2[i])
        print("\nBEFORE LoRA (baseline):")
        print(base_type2[i])
        print("\nAFTER LoRA (styled):")
        print(out_type2[i])
