from collections import Counter
from pathlib import Path
import re

import matplotlib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.load_data import load_dataset


matplotlib.use("Agg")
from matplotlib import pyplot as plt


REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
RANDOM_STATE = 42
TEST_SIZE = 0.2
ELONGATED_WORD_PATTERN = re.compile(
    r"([a-z])\1{2,}",
    flags=re.IGNORECASE,
)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\U00002600-\U000026FF"
    "\U00002700-\U000027BF"
    "]"
)
UNKNOWN_VALUES = {"unknown", "null", "none", "nan", "n/a", "na", "-", "?"}
STOPWORDS = {
    "ada", "akan", "aku", "anda", "bisa", "dan", "dari", "di", "dengan",
    "ini", "itu", "jadi", "juga", "kalau", "kami", "ke", "karena", "nya",
    "pada", "saja", "sama", "saya", "sebagai", "sudah", "tapi", "untuk", "yang",
}


def build_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Menghitung temuan kualitas data tanpa mengubah DataFrame asli."""
    text = df["text"].fillna("").astype(str)
    stripped = text.str.strip()

    duplicate_text = df.duplicated(subset=["text"], keep=False)
    label_counts = df.groupby("text", dropna=False)["label"].nunique(dropna=False)
    conflicting_texts = label_counts[label_counts > 1].index
    conflicting = df["text"].isin(conflicting_texts)

    issues = {
        "missing_text": int(df["text"].isna().sum()),
        "missing_label": int(df["label"].isna().sum()),
        "blank_or_whitespace_text": int(stripped.eq("").sum()),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_text_rows": int(duplicate_text.sum()),
        "duplicate_text_groups": int(
            df.loc[duplicate_text, "text"].nunique(dropna=False)
        ),
        "conflicting_label_rows": int(conflicting.sum()),
        "conflicting_text_groups": int(len(conflicting_texts)),
        "leading_or_trailing_space": int(text.ne(stripped).sum()),
        "repeated_whitespace": int(text.str.contains(r"\s{2,}").sum()),
        "unknown_like_text": int(stripped.str.lower().isin(UNKNOWN_VALUES).sum()),
        "placeholder_token": int(
            text.str.contains(r"__[a-z_]+__", case=False).sum()
        ),
        "elongated_word_rows": int(
            text.map(
                lambda value: bool(ELONGATED_WORD_PATTERN.search(value))
            ).sum()
        ),
        "very_short_text_le_2_words": int(
            stripped.str.split().str.len().le(2).sum()
        ),
        "contains_url": int(
            text.str.contains(r"https?://|www\.", case=False).sum()
        ),
        "contains_at_mention": int(
            text.str.contains(r"(?<!\w)@\w+").sum()
        ),
        "contains_hashtag": int(
            text.str.contains(r"(?<!\w)#\w+").sum()
        ),
        "contains_emoji": int(
            text.map(
                lambda value: bool(EMOJI_PATTERN.search(value))
            ).sum()
        ),
    }

    return pd.DataFrame(issues.items(), columns=["issue", "count"])


def prepare_modeling_data(df: pd.DataFrame) -> pd.DataFrame:
    """Menghapus exact duplicate sebelum split agar tidak bocor antarpartisi."""
    return (
        df.drop_duplicates(subset=["text", "label"])
        .reset_index(drop=True)
    )


def split_modeling_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Membuat split kanonik yang dipakai EDA, training, dan evaluasi."""
    modeling_df = prepare_modeling_data(df)

    train_df, test_df = train_test_split(
        modeling_df,
        test_size=TEST_SIZE,
        stratify=modeling_df["label"],
        random_state=RANDOM_STATE,
    )

    return train_df, test_df


def save_eda_plots(
    train_df: pd.DataFrame,
    output_dir: Path = REPORTS_DIR,
    quality_df: pd.DataFrame | None = None,
) -> list[Path]:
    """Menyimpan empat grafik EDA wajib ke direktori reports/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "label_distribution.png",
        output_dir / "missing_values_by_column.png",
        output_dir / "text_length_by_label.png",
        output_dir / "top_terms_by_label.png",
    ]

    # 1. Sebaran target. Hanya train set yang dipakai.
    label_counts = train_df["label"].value_counts().sort_index()
    ax = label_counts.plot.bar(
        color="#4472C4",
        title="Distribusi Label Sentimen pada Train Set",
    )
    ax.set(xlabel="Label", ylabel="Jumlah teks")
    ax.tick_params(axis="x", rotation=0)
    ax.figure.tight_layout()
    ax.figure.savefig(paths[0], dpi=150)
    plt.close(ax.figure)

    # 2. Nilai hilang per kolom sesuai pertanyaan wajib pada lembar UAS.
    audit_df = train_df if quality_df is None else quality_df
    missing_counts = audit_df.isna().sum().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(missing_counts.index.astype(str), missing_counts.values)
    ax.set(
        title="Jumlah Nilai Hilang per Kolom",
        xlabel="Jumlah nilai hilang",
        ylabel="Kolom",
    )

    max_missing = int(missing_counts.max())
    ax.set_xlim(0, max(1, max_missing * 1.1))

    for position, count in enumerate(missing_counts.values):
        ax.text(float(count), position, f" {int(count)}", va="center")
    fig.tight_layout()
    fig.savefig(paths[1], dpi=150)
    plt.close(fig)

    # 3. Panjang teks per kelas. Hanya train set yang dipakai.
    labels = label_counts.index.tolist()
    lengths = [
        train_df.loc[train_df["label"].eq(label), "text"]
        .fillna("")
        .str.split()
        .str.len()
        for label in labels
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(lengths, tick_labels=labels, showfliers=False)
    ax.set(
        title="Panjang Teks per Label pada Train Set",
        xlabel="Label",
        ylabel="Jumlah kata",
    )
    fig.tight_layout()
    fig.savefig(paths[2], dpi=150)
    plt.close(fig)

    # 4. Ciri linguistik tiap kelas tanpa stopword. Hanya train set yang dipakai.
    fig, axes = plt.subplots(len(labels), 1, figsize=(8, 3 * len(labels)))
    axes = [axes] if len(labels) == 1 else axes

    for ax, label in zip(axes, labels):
        combined_text = " ".join(
            train_df.loc[train_df["label"].eq(label), "text"].fillna("")
        )
        terms = Counter(
            word
            for word in re.findall(r"\b[a-z]{2,}\b", combined_text.lower())
            if word not in STOPWORDS
        ).most_common(10)

        words, counts = zip(*terms) if terms else ([], [])
        ax.barh(words[::-1], counts[::-1], color="#70AD47")
        ax.set(
            title=f"Istilah Teratas: {label}",
            xlabel="Frekuensi",
        )

    fig.tight_layout()
    fig.savefig(paths[3], dpi=150)
    plt.close(fig)

    return paths


def main() -> None:
    df = load_dataset()

    print("\n=== DF.ISNA().SUM() ===")
    print(df.isna().sum().to_string())

    print("\n=== DF.DESCRIBE() ===")
    print(df.describe(include="all").to_string())

    print("\n=== DISTRIBUSI TARGET ===")
    print(df["label"].value_counts(dropna=False).to_string())

    print("\n=== PEMERIKSAAN DUPLIKAT ===")
    print(f"Exact duplicate rows: {int(df.duplicated().sum())}")

    print("\n=== AUDIT KEKOTORAN DATA ===")
    quality_summary = build_quality_summary(df)
    print(quality_summary.to_string(index=False))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / "data_quality_summary.csv"
    quality_summary.to_csv(output_path, index=False)

    issue_counts = quality_summary.set_index("issue")["count"].to_dict()
    fatal_issue_names = [
        "missing_text",
        "missing_label",
        "blank_or_whitespace_text",
        "conflicting_text_groups",
    ]
    fatal_issues = {
        issue: issue_counts[issue]
        for issue in fatal_issue_names
        if issue_counts[issue] > 0
    }

    if fatal_issues:
        raise ValueError(
            "Split dibatalkan karena masalah data fatal yang harus dikurasi: "
            f"{fatal_issues}"
        )

    train_df, test_df = split_modeling_data(df)
    plot_paths = save_eda_plots(train_df, quality_df=df)

    print("\n=== SPLIT MODELING ===")
    print(f"Jumlah data setelah deduplikasi: {len(train_df) + len(test_df)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows terkunci: {len(test_df)}")

    print(f"\nRingkasan audit disimpan ke:\n{output_path}")
    print("\nGrafik EDA:\n" + "\n".join(map(str, plot_paths)))


if __name__ == "__main__":
    main()
