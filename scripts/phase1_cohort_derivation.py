"""
Revizyon Phase 1 - Kohort Yeniden Tanimi
=================================================================================
Hakem 2'nin en kritik itirazi: reported PSAD'in complete-case sarti olarak
kullanilmasi (a) Model A/A' icin gereksiz seciim yanliligi yaratiyor (bu
modeller PSA+hacim kullaniyor, PSAD degil), (b) reported-PSAD-vs-AI-PSAD
karsilastirmasini "AI hacim etkisi" ile "reported PSAD alanindaki veri
kalitesi sorunlari (transkripsiyon/rounding/birim hatasi)" karistiran
gecersiz bir karsilastirmaya ceviriyor.

Bu script iki kohortu ayri ayri, reported-PSAD SARTI OLMADAN kuruyor:
  - Cohort A: single-marker PSAD karsilastirmasi icin (calculated manual
    PSAD = PSA/manuel-hacim, vs AI-PSAD = PSA/AI-hacim)
  - Cohort B: Model A / Model A' multivariable karsilastirmasi icin

Reported PSAD sadece Adim 4'te (discordance analizi) kullaniliyor -- artik
PRIMARY comparator degil, SECONDARY/veri-kalitesi analizi.

Girdi: Faz 2'nin kaydettigi faz2_merged_clinical_ai_volume.csv. Bu dosya
ZATEN patient-level dedup edilmis (1476 hasta) ve known-faulty segmentasyon
(11050_1001070) + implausible (<5mL) AI hacimler ZATEN NaN'a cevrilmis
(bir onceki, artik supersede edilmis faz'da: flag_implausible_volumes,
exclude_known_faulty_cases). Bu script o islemleri TEKRAR YAPMIYOR, sadece
denetliyor (audit) ve uzerine kohort tanimlarini kuruyor.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

DATASET_ROOT = "/content/drive/MyDrive/Uroloji_Projesi/PICAI_Dataset"
MERGED_CSV_PATH = os.path.join(DATASET_ROOT, "QA_outputs", "faz2_merged_clinical_ai_volume.csv")
OUTPUT_DIR = os.path.join(DATASET_ROOT, "QA_outputs", "revision_outputs")

RANDOM_SEED = 42  # sonraki fazlardaki bootstrap/CV icin sabit, burada kullanilmiyor
KNOWN_FAULTY_CASES = {"11050_1001070"}
EXPECTED_N_PATIENTS = 1476


def print_header(title):
    print("\n" + "=" * 90)
    print(f"||  {title}")
    print("=" * 90)


def print_subheader(title):
    print(f"\n{'-' * 80}\n  {title}\n{'-' * 80}")


# ============================================================
# STEP 1: MASTER ANALYTIC DATAFRAME
# ============================================================
def build_master_dataframe(csv_path):
    print_header("STEP 1: MASTER ANALYTIC DATAFRAME")
    df = pd.read_csv(csv_path)
    print(f"Girdi (faz2_merged_clinical_ai_volume.csv): {len(df)} satir")

    if len(df) != EXPECTED_N_PATIENTS:
        print(f"UYARI: beklenen {EXPECTED_N_PATIENTS} hasta degil, {len(df)} -- "
              f"CSV Faz 2'nin beklenen ciktisi olmayabilir, kontrol et.")

    master = pd.DataFrame({
        'patient_id': df['patient_id'],
        'case_id': df['case_id'],
        'center': df['center'],
        'age': df['patient_age'],
        'psa': df['psa'],
        'manual_volume': df['prostate_volume'],
        'reported_psad': df['psad'],
        'ai_volume_alg1': df['ai_volume_Bosma22b'],  # Faz 2'de zaten NaN'landi (fault+implausible)
        'ai_volume_alg2': df['ai_volume_Guerbet23'],  # ayni sekilde zaten NaN'landi
        'csPCa_status': (df['case_csPCa'] == 'YES').astype(int),
        'isup_grade': df['case_ISUP'],
    })

    # Audit: known-faulty vaka gercekten NaN mi? (Faz 2'nin islemini TEKRARLAMIYORUZ, dogruluyoruz)
    for cid in KNOWN_FAULTY_CASES:
        row = master.loc[master['case_id'] == cid]
        if len(row) == 0:
            print(f"UYARI: bilinen hatali vaka {cid} master dataframe'de bulunamadi.")
        else:
            val = row['ai_volume_alg1'].iloc[0]
            status = "NaN (dogru, disarida)" if pd.isna(val) else f"DOLU ({val:.1f} mL) -- BEKLENMIYORDU"
            print(f"Audit -- bilinen hatali vaka {cid}: ai_volume_alg1 = {status}")

    print(f"\nMissingness (master dataframe, N={len(master)}):")
    for col in ['age', 'psa', 'manual_volume', 'reported_psad', 'ai_volume_alg1', 'ai_volume_alg2', 'center']:
        n_missing = master[col].isna().sum()
        print(f"  {col:<20}: {n_missing} eksik ({100 * n_missing / len(master):.1f}%)")

    return master


# ============================================================
# STEP 2.2 (kullanici siralamasinda "Cohort B"): MULTIVARIABLE MODEL COHORT
# ============================================================
def define_cohort_b_model(master):
    print_header("COHORT B — MULTIVARIABLE MODEL COHORT (Model A / Model A')")
    print("Kriter: age + PSA + manual_volume + AI_volume(alg1) + center + csPCa mevcut, "
          "PCNN haric. reported_psad ZORUNLU DEGIL.")

    mask = (
        master['age'].notna() &
        master['psa'].notna() &
        master['manual_volume'].notna() &
        master['ai_volume_alg1'].notna() &
        master['center'].notna() &
        master['csPCa_status'].notna() &
        (master['center'] != 'PCNN')
    )
    cohort_b = master[mask].copy()

    n = len(cohort_b)
    n_pos = int(cohort_b['csPCa_status'].sum())
    print(f"\nN = {n}")
    print(f"csPCa+ = {n_pos} ({100 * n_pos / n:.1f}%)")
    print("\nCenter dagilimi:")
    for c in sorted(cohort_b['center'].unique()):
        sub = cohort_b[cohort_b['center'] == c]
        print(f"  {c}: N={len(sub)} ({100 * len(sub) / n:.1f}%), "
              f"csPCa+={int(sub['csPCa_status'].sum())} ({100 * sub['csPCa_status'].mean():.1f}%)")

    n_with_reported_psad = cohort_b['reported_psad'].notna().sum()
    print(f"\nBilgi amacli: Cohort B'nin {n_with_reported_psad}/{n} hastasinda reported_psad DA mevcut "
          f"(kriter olarak kullanilmadi, sadece referans).")

    return cohort_b


# ============================================================
# STEP 2.1 ("Cohort A"): SINGLE-MARKER PSAD COHORT
# ============================================================
def define_cohort_a_single_marker(master):
    print_header("COHORT A — SINGLE-MARKER PSAD COHORT (calculated manual PSAD vs AI-PSAD)")
    print("Kriter: PSA + manual_volume + AI_volume(alg1) + csPCa mevcut.")
    print("ONEMLI KARAR: age ve center BURADA ZORUNLU DEGIL -- bu karsilastirmada")
    print("kovaryat olarak kullanilmiyorlar (sadece PSA/hacim orani, iki marker).")
    print("Bu, kullanicinin kisa Phase-1 notundaki 'Cohort B ile ayni kriter' ifadesinden")
    print("BILEREK farkli bir yorum: hakemin 'gereksiz complete-case sarti seciim yanliligi")
    print("yaratir' itirazini age/center icin de ayni mantikla uyguluyoruz -- bu ikisi")
    print("single-marker karsilastirmada kullanilmadigi icin zorunlu tutmak gereksiz N kaybi olur.")

    mask = (
        master['psa'].notna() &
        master['manual_volume'].notna() &
        master['ai_volume_alg1'].notna() &
        master['csPCa_status'].notna()
    )
    cohort_a = master[mask].copy()

    n = len(cohort_a)
    n_pos = int(cohort_a['csPCa_status'].sum())
    print(f"\nN = {n} (age/center sarti YOK)")
    print(f"csPCa+ = {n_pos} ({100 * n_pos / n:.1f}%)")

    n_center_missing = cohort_a['center'].isna().sum()
    n_pcnn = (cohort_a['center'] == 'PCNN').sum()
    print(f"\nBilgi amacli center dagilimi (PCNN BU KOHORTTAN DISLANMADI, cunku merkez "
          f"burada kovaryat degil):")
    for c in sorted(cohort_a['center'].dropna().unique()):
        sub = cohort_a[cohort_a['center'] == c]
        print(f"  {c}: N={len(sub)}")
    if n_center_missing > 0:
        print(f"  (center eksik: {n_center_missing})")
    print(f"  -- bunun icinde PCNN: {n_pcnn}")

    return cohort_a


def report_overlap(cohort_a, cohort_b):
    print_header("COHORT A vs COHORT B ORTUSUM")
    ids_a = set(cohort_a['case_id'])
    ids_b = set(cohort_b['case_id'])
    overlap = ids_a & ids_b
    only_a = ids_a - ids_b
    only_b = ids_b - ids_a

    print(f"Cohort A N = {len(ids_a)}")
    print(f"Cohort B N = {len(ids_b)}")
    print(f"Kesisim N = {len(overlap)}")
    print(f"Sadece Cohort A'da (B'de degil -- muhtemelen age/center/PCNN nedeniyle) N = {len(only_a)}")
    print(f"Sadece Cohort B'de (A'da degil) N = {len(only_b)}")
    if len(only_b) > 0:
        print("  UYARI: Cohort B'nin tum kriterleri Cohort A'nin kriterlerini de sagliyor olmali "
              "(B daha kisitli). Bu deger 0 degilse mantik hatasi var, kontrol et.")
    else:
        print("  Beklenen sonuc dogrulandi: Cohort B, Cohort A'nin bir alt kumesi.")


# ============================================================
# STEP 4: REPORTED PSAD DISCORDANCE ANALIZI (secondary/veri-kalitesi)
# ============================================================
def discordance_analysis(cohort_a):
    print_header("STEP 4: REPORTED PSAD DISCORDANCE ANALIZI (secondary, ARTIK PRIMARY DEGIL)")

    sub = cohort_a.dropna(subset=['reported_psad']).copy()
    sub['calculated_manual_psad'] = sub['psa'] / sub['manual_volume']

    n = len(sub)
    print(f"Hem reported hem calculated PSAD mevcut olan N = {n} (Cohort A'nin alt kumesi, "
          f"Cohort A N={len(cohort_a)})")

    r, p_r = stats.pearsonr(sub['reported_psad'], sub['calculated_manual_psad'])
    rho, p_rho = stats.spearmanr(sub['reported_psad'], sub['calculated_manual_psad'])
    abs_diff = (sub['reported_psad'] - sub['calculated_manual_psad']).abs()
    mad = abs_diff.mean()
    median_ad = abs_diff.median()

    print(f"\nPearson r  = {r:.4f} (p={p_r:.3g})")
    print(f"Spearman rho = {rho:.4f} (p={p_rho:.3g})")
    print(f"Mean absolute difference   = {mad:.4f}")
    print(f"Median absolute difference = {median_ad:.4f}")

    ratio = sub['reported_psad'] / sub['calculated_manual_psad']
    fold2_mask = (ratio > 2) | (ratio < 0.5)
    fold3_mask = (ratio > 3) | (ratio < 1 / 3)
    n2, n3 = int(fold2_mask.sum()), int(fold3_mask.sum())
    print(f"\n>2-fold discordant: {n2} ({100 * n2 / n:.1f}%)")
    print(f">3-fold discordant: {n3} ({100 * n3 / n:.1f}%)")

    discordant_cases = pd.DataFrame()
    if n3 > 0:
        print("\n>3-fold discordant vakalarin ozellikleri:")
        cols = ['case_id', 'center', 'age', 'psa', 'manual_volume', 'reported_psad',
                'calculated_manual_psad', 'ai_volume_alg1', 'csPCa_status', 'isup_grade']
        discordant_cases = sub.loc[fold3_mask, cols].copy()
        discordant_cases['ratio_reported_over_calculated'] = ratio[fold3_mask].values
        print(discordant_cases.to_string(index=False))
    else:
        print("\n>3-fold discordant vaka yok.")

    return sub, discordant_cases


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    master = build_master_dataframe(MERGED_CSV_PATH)
    cohort_b = define_cohort_b_model(master)
    cohort_a = define_cohort_a_single_marker(master)
    report_overlap(cohort_a, cohort_b)
    discordance_sub, discordant_cases = discordance_analysis(cohort_a)

    cohort_b.to_csv(os.path.join(OUTPUT_DIR, "cohort_b_model.csv"), index=False)
    cohort_a.to_csv(os.path.join(OUTPUT_DIR, "cohort_a_single_marker.csv"), index=False)
    discordance_sub.to_csv(os.path.join(OUTPUT_DIR, "supp_reported_psad_discordance.csv"), index=False)
    if len(discordant_cases) > 0:
        discordant_cases.to_csv(os.path.join(OUTPUT_DIR, "supp_reported_psad_discordant_cases.csv"), index=False)

    summary = pd.DataFrame([
        {'cohort': 'Cohort A (single-marker PSAD)', 'n': len(cohort_a),
         'cspca_pos': int(cohort_a['csPCa_status'].sum()),
         'cspca_pct': round(100 * cohort_a['csPCa_status'].mean(), 1)},
        {'cohort': 'Cohort B (multivariable model)', 'n': len(cohort_b),
         'cspca_pos': int(cohort_b['csPCa_status'].sum()),
         'cspca_pct': round(100 * cohort_b['csPCa_status'].mean(), 1)},
    ])
    summary.to_csv(os.path.join(OUTPUT_DIR, "cohort_derivation_summary.csv"), index=False)

    print_header("PHASE 1 TAMAMLANDI — OZET")
    print(summary.to_string(index=False))
    print(f"\nCiktilar kaydedildi: {OUTPUT_DIR}")
    print("  - cohort_b_model.csv")
    print("  - cohort_a_single_marker.csv")
    print("  - supp_reported_psad_discordance.csv")
    print("  - supp_reported_psad_discordant_cases.csv (varsa)")
    print("  - cohort_derivation_summary.csv")


if __name__ == "__main__":
    main()
