"""
Targeted verification -- Results 3.2 Bland-Altman sayilarini (mean bias, 95% LoA, CCC)
dogrulamak icin. Tam pipeline (Phase 7) yeniden calistirilmiyor, sadece bu iki sayi
icin gerekli minimal hesap yapiliyor.

Kohort: manual_volume (clinically reported) VE ai_volume_Bosma22b (Algorithm 1) her
ikisi de mevcut, bilinen hatali vaka (11050_1001070) haric -- Results 3.2'de
"n=1449" olarak belirtilen kohortla ayni tanim.
"""

import os
import numpy as np
import pandas as pd

DATASET_ROOT = "/content/drive/MyDrive/Uroloji_Projesi/PICAI_Dataset"
MERGED_CSV_PATH = os.path.join(DATASET_ROOT, "QA_outputs", "faz2_merged_clinical_ai_volume.csv")
KNOWN_FAULTY_CASES = {"11050_1001070"}


def concordance_correlation_coefficient(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(ddof=0), y.var(ddof=0)
    cov = np.mean((x - mx) * (y - my))
    return (2 * cov) / (vx + vy + (mx - my) ** 2)


def main():
    df = pd.read_csv(MERGED_CSV_PATH)

    valid = df[
        df['prostate_volume'].notna()
        & df['ai_volume_Bosma22b'].notna()
        & ~df['case_id'].astype(str).isin(KNOWN_FAULTY_CASES)
    ].copy()

    n = len(valid)
    manual = valid['prostate_volume'].to_numpy(dtype=float)
    ai = valid['ai_volume_Bosma22b'].to_numpy(dtype=float)

    diff = ai - manual  # AI - clinical (metindeki "AI-derived volume was lower than
                         # clinical by mean of X" cumlesiyle ayni yon)
    mean_diff = diff.mean()
    sd_diff = diff.std(ddof=1)
    loa_lower = mean_diff - 1.96 * sd_diff
    loa_upper = mean_diff + 1.96 * sd_diff
    ccc_val = concordance_correlation_coefficient(manual, ai)

    print("=" * 70)
    print("RESULTS 3.2 -- Bland-Altman dogrulama (AI-derived vs clinically reported)")
    print("=" * 70)
    print(f"N (valid pairs, known-faulty case excluded) = {n}")
    print(f"Mean difference (AI - clinical)            = {mean_diff:.4f} mL")
    print(f"SD of differences                          = {sd_diff:.4f} mL")
    print(f"95% Limits of Agreement                    = [{loa_lower:.4f}, {loa_upper:.4f}] mL")
    print(f"Concordance correlation coefficient (CCC)  = {ccc_val:.4f}")
    print()
    print("Metindeki mevcut iddia: n=1449, mean bias=-5.5 mL, LoA=[-35.9, 24.9] mL, CCC=0.888")
    if n != 1449:
        print(f">>> UYARI: N eslesmiyor ({n} vs beklenen 1449) -- kohort tanimi farkli olabilir.")


if __name__ == "__main__":
    main()
