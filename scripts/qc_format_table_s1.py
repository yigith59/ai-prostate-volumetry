"""
Supplementary Table S1 - Full Visual QC Case List (25 cases)
=============================================================
Amac: qc_visual_review.py'nin kaydettigi qa_candidate_list.csv'yi
okuyup, manuscript'e dogrudan yapistirilabilecek, EKSIKSIZ (tum 25 vaka,
Algorithm 2 hacimleri dahil) ve okunakli bir Supplementary Table S1
metni uretir. Elle transkripsiyon / placeholder riski ortadan kalkar.
"""

import os
import pandas as pd

DATASET_ROOT = "/content/drive/MyDrive/Uroloji_Projesi/PICAI_Dataset"
QA_CSV_PATH = os.path.join(DATASET_ROOT, "QA_outputs", "faz2b_volume_qa", "qa_candidate_list.csv")

REASON_LABELS = {
    "bilinen_hatali_segmentasyon": "Known segmentation error (curator-documented)",
    "yuksek_oran_sapmasi": "High discordance (ratio {r})",
    "orta_oran_sapmasi": "Moderate discordance (ratio {r})",
    "AI_hacim_implausible": "Physiologically implausible AI volume",
    "AI_algoritmalari_uyumsuz": "Inter-algorithm disagreement (ratio {r})",
    "rastgele_kontrol_ornegi": "Random control sample",
}

# Ilk eslesen anahtar en yuksek onceligi belirler (siralama icin)
PRIORITY_ORDER = [
    "bilinen_hatali_segmentasyon",
    "yuksek_oran_sapmasi",
    "orta_oran_sapmasi",
    "AI_hacim_implausible",
    "AI_algoritmalari_uyumsuz",
    "rastgele_kontrol_ornegi",
]


def translate_reasons(raw):
    parts = [p.strip() for p in raw.split(";") if p.strip()]
    translated = []
    top_priority = len(PRIORITY_ORDER)
    for p in parts:
        if "(" in p:
            key, rest = p.split("(", 1)
            ratio_val = rest.rstrip(")")
        else:
            key, ratio_val = p, None
        label = REASON_LABELS.get(key, key)
        if "{r}" in label and ratio_val:
            label = label.format(r=ratio_val)
        translated.append(label)
        if key in PRIORITY_ORDER:
            top_priority = min(top_priority, PRIORITY_ORDER.index(key))
    return "; ".join(translated), top_priority


def main():
    df = pd.read_csv(QA_CSV_PATH)
    df["flagging_criterion"], df["_priority"] = zip(*df["qa_reasons"].map(translate_reasons))
    df["ratio_alg1_manual"] = df["ai_volume_Bosma22b"] / df["prostate_volume"]
    df = df.sort_values(["_priority", "case_id"]).reset_index(drop=True)

    print(f"Toplam vaka: {len(df)}\n")
    header = f"{'Case ID':<16} | {'Manual (mL)':>11} | {'AI Alg1 (mL)':>12} | {'AI Alg2 (mL)':>12} | {'Ratio (Alg1/manual)':>20} | Flagging criterion"
    print(header)
    print("-" * len(header))
    for _, row in df.iterrows():
        manual = f"{row['prostate_volume']:.1f}" if pd.notna(row['prostate_volume']) else "—"
        alg1 = f"{row['ai_volume_Bosma22b']:.1f}" if pd.notna(row['ai_volume_Bosma22b']) else "—"
        alg2 = f"{row['ai_volume_Guerbet23']:.1f}" if pd.notna(row['ai_volume_Guerbet23']) else "—"
        ratio = f"{row['ratio_alg1_manual']:.2f}" if pd.notna(row['ratio_alg1_manual']) else "—"
        print(f"{row['case_id']:<16} | {manual:>11} | {alg1:>12} | {alg2:>12} | {ratio:>20} | {row['flagging_criterion']}")

    print("\n--- Markdown pipe-table (manuscript'e yapistirmaya hazir) ---\n")
    print("| Case ID | Manual volume, mL | AI volume Alg 1, mL | AI volume Alg 2, mL | Ratio (Alg1/manual) | Flagging criterion |")
    print("|---|---|---|---|---|---|")
    for _, row in df.iterrows():
        manual = f"{row['prostate_volume']:.1f}" if pd.notna(row['prostate_volume']) else "—"
        alg1 = f"{row['ai_volume_Bosma22b']:.1f}" if pd.notna(row['ai_volume_Bosma22b']) else "—"
        alg2 = f"{row['ai_volume_Guerbet23']:.1f}" if pd.notna(row['ai_volume_Guerbet23']) else "—"
        ratio = f"{row['ratio_alg1_manual']:.2f}" if pd.notna(row['ratio_alg1_manual']) else "—"
        print(f"| {row['case_id']} | {manual} | {alg1} | {alg2} | {ratio} | {row['flagging_criterion']} |")


if __name__ == "__main__":
    main()
