"""
Revizyon Phase 1B - PCNN Denetimi + Cohort A Sensitivity + Discordance Sensitivity
=================================================================================
Iki farkli AI'in Phase 1 sonuclarini degerlendirmesinden cikan ortak talep:
PCNN'in Cohort B'den (Model A/A') disariya alinmasi hala "yetersiz veri"
mi, yoksa artik sadece bir TASARIM TERCIHI mi -- bunu KANITLA, varsayma.

Bu script uc ayri seyi yapiyor:
1. PCNN eligibility audit: PCNN'deki kac hasta, center disinda Cohort B'nin
   TUM diger kriterlerini (age+PSA+manual_volume+AI_volume+csPCa) karsiliyor?
   Eger bu sayi Cohort A'daki PCNN sayisiyla (322) ayniysa, PCNN'nin
   disarida kalmasinin nedeni eksik veri DEGIL, sadece explicit filtre --
   bu, iki AI'nin da "kirmizi bayrak" dedigi noktanin dogru olup olmadigini
   gosterir.
2. Cohort A sensitivity (RUMC/ZGT-only): PCNN'siz versiyon, Cohort B ile
   ayni merkez kompozisyonunda single-marker karsilastirma yapilabilsin.
3. Discordance sensitivity: >3-fold discordant 12 vaka CIKARILINCA
   Pearson r / Spearman rho / MAD nasil degisiyor -- bu, "moderate Pearson
   korelasyonu bir avuc asiri deger tarafindan suruklenıyor" iddiasini
   sayisal olarak kanitliyor.

Girdi: ayni faz2_merged_clinical_ai_volume.csv (Phase 1 ile birebir ayni
mantikla master dataframe kuruluyor, self-contained tutmak icin tekrarlandi).
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


def print_header(title):
    print("\n" + "=" * 90)
    print(f"||  {title}")
    print("=" * 90)


def build_master_dataframe(csv_path):
    df = pd.read_csv(csv_path)
    master = pd.DataFrame({
        'patient_id': df['patient_id'],
        'case_id': df['case_id'],
        'center': df['center'],
        'age': df['patient_age'],
        'psa': df['psa'],
        'manual_volume': df['prostate_volume'],
        'reported_psad': df['psad'],
        'ai_volume_alg1': df['ai_volume_Bosma22b'],
        'ai_volume_alg2': df['ai_volume_Guerbet23'],
        'csPCa_status': (df['case_csPCa'] == 'YES').astype(int),
        'isup_grade': df['case_ISUP'],
    })
    return master


# ============================================================
# 1. PCNN ELIGIBILITY AUDIT
# ============================================================
def pcnn_eligibility_audit(master):
    print_header("1. PCNN ELIGIBILITY AUDIT — 'yetersiz veri' mi 'tasarim tercihi' mi?")

    # Cohort B'nin TUM kriterleri, AMA center!='PCNN' filtresi OLMADAN
    mask_all_criteria_any_center = (
        master['age'].notna() &
        master['psa'].notna() &
        master['manual_volume'].notna() &
        master['ai_volume_alg1'].notna() &
        master['center'].notna() &
        master['csPCa_status'].notna()
    )
    cohort_b_3centre = master[mask_all_criteria_any_center].copy()

    n_total = len(cohort_b_3centre)
    print(f"Cohort B kriterleri (age+PSA+manual_vol+AI_vol+center+csPCa mevcut), "
          f"PCNN filtresi OLMADAN: N = {n_total}")

    for c in sorted(cohort_b_3centre['center'].unique()):
        sub = cohort_b_3centre[cohort_b_3centre['center'] == c]
        n_events = int(sub['csPCa_status'].sum())
        print(f"  {c}: N={len(sub)}, csPCa+={n_events} ({100 * sub['csPCa_status'].mean():.1f}%)")

    n_pcnn_eligible = int((cohort_b_3centre['center'] == 'PCNN').sum())
    n_pcnn_in_cohort_a = 322  # Phase 1'de raporlanan, karsilastirma icin sabit referans

    print(f"\nPCNN'de Cohort B'nin TUM diger kriterlerini (age/PSA/manual/AI/csPCa) karsilayan "
          f"hasta sayisi: {n_pcnn_eligible}")
    print(f"Phase 1'de Cohort A'da bulunan PCNN sayisi (referans): {n_pcnn_in_cohort_a}")

    if n_pcnn_eligible == n_pcnn_in_cohort_a:
        print("\n>>> SONUC: Sayilar BIREBIR AYNI. Bu, PCNN'nin Cohort B'den disarida kalmasinin "
              "'yetersiz veri' NEDENIYLE OLMADIGINI kanitliyor -- tek neden, explicit "
              "'center != PCNN' filtresi. Bu bir TASARIM TERCIHI, veri kisiti degil. "
              "Methods'ta bu ayrimin acikca boyle yazilmasi gerekiyor.")
    else:
        diff = n_pcnn_in_cohort_a - n_pcnn_eligible
        print(f"\n>>> SONUC: Sayilar FARKLI (fark={diff}). PCNN'nin bir kismi gercekten "
              f"age/center gibi ek bir degiskende eksik veriye sahip -- disarida kalma nedeni "
              f"kismen veri kisiti, kismen tasarim tercihi. Detaylandirilmali.")

    return cohort_b_3centre


# ============================================================
# 2. COHORT A SENSITIVITY (RUMC/ZGT-only)
# ============================================================
def cohort_a_rumc_zgt_sensitivity(master):
    print_header("2. COHORT A SENSITIVITY — RUMC/ZGT-only (PCNN haric)")

    mask = (
        master['psa'].notna() &
        master['manual_volume'].notna() &
        master['ai_volume_alg1'].notna() &
        master['csPCa_status'].notna() &
        (master['center'].isin(['RUMC', 'ZGT']))
    )
    cohort_a_sens = master[mask].copy()

    n = len(cohort_a_sens)
    n_pos = int(cohort_a_sens['csPCa_status'].sum())
    print(f"N = {n} (Cohort A'nin PCNN'siz hali)")
    print(f"csPCa+ = {n_pos} ({100 * n_pos / n:.1f}%)")
    for c in sorted(cohort_a_sens['center'].unique()):
        sub = cohort_a_sens[cohort_a_sens['center'] == c]
        print(f"  {c}: N={len(sub)}, csPCa+={int(sub['csPCa_status'].sum())} "
              f"({100 * sub['csPCa_status'].mean():.1f}%)")

    return cohort_a_sens


# ============================================================
# 3. DISCORDANCE SENSITIVITY (12 aykiri vaka cikarilinca)
# ============================================================
def discordance_sensitivity(master):
    print_header("3. DISCORDANCE SENSITIVITY — >3-fold aykiri 12 vaka cikarilinca")

    cohort_a_mask = (
        master['psa'].notna() &
        master['manual_volume'].notna() &
        master['ai_volume_alg1'].notna() &
        master['csPCa_status'].notna()
    )
    cohort_a = master[cohort_a_mask].copy()

    sub = cohort_a.dropna(subset=['reported_psad']).copy()
    sub['calculated_manual_psad'] = sub['psa'] / sub['manual_volume']
    ratio = sub['reported_psad'] / sub['calculated_manual_psad']
    fold3_mask = (ratio > 3) | (ratio < 1 / 3)

    n_before = len(sub)
    r_before, p_before = stats.pearsonr(sub['reported_psad'], sub['calculated_manual_psad'])
    rho_before, _ = stats.spearmanr(sub['reported_psad'], sub['calculated_manual_psad'])
    mad_before = (sub['reported_psad'] - sub['calculated_manual_psad']).abs().mean()
    median_ad_before = (sub['reported_psad'] - sub['calculated_manual_psad']).abs().median()

    sub_excl = sub[~fold3_mask].copy()
    n_after = len(sub_excl)
    r_after, p_after = stats.pearsonr(sub_excl['reported_psad'], sub_excl['calculated_manual_psad'])
    rho_after, _ = stats.spearmanr(sub_excl['reported_psad'], sub_excl['calculated_manual_psad'])
    mad_after = (sub_excl['reported_psad'] - sub_excl['calculated_manual_psad']).abs().mean()
    median_ad_after = (sub_excl['reported_psad'] - sub_excl['calculated_manual_psad']).abs().median()

    print(f"{'Metrik':<30} | {'Oncesi (N=' + str(n_before) + ')':>20} | {'Sonrasi (N=' + str(n_after) + ')':>20}")
    print("-" * 76)
    print(f"{'Pearson r':<30} | {r_before:>20.4f} | {r_after:>20.4f}")
    print(f"{'Spearman rho':<30} | {rho_before:>20.4f} | {rho_after:>20.4f}")
    print(f"{'Mean absolute difference':<30} | {mad_before:>20.4f} | {mad_after:>20.4f}")
    print(f"{'Median absolute difference':<30} | {median_ad_before:>20.4f} | {median_ad_after:>20.4f}")

    delta_r = r_after - r_before
    print(f"\nPearson r'deki degisim: {delta_r:+.4f}")
    if delta_r > 0.1:
        print(">>> SONUC: 12 vakanin cikarilmasi Pearson r'yi belirgin sekilde yukseltiyor -- "
              "bu, 'moderate Pearson korelasyonu az sayida asiri deger tarafindan suruklendi' "
              "iddiasini sayisal olarak DOGRULUYOR.")
    else:
        print(">>> SONUC: 12 vakanin cikarilmasi Pearson r'yi cok degistirmiyor -- discordance "
              "daha genis-tabanli, sadece bu 12 vakaya indirgenemez.")

    return sub, sub_excl


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    master = build_master_dataframe(MERGED_CSV_PATH)

    cohort_b_3centre = pcnn_eligibility_audit(master)
    cohort_a_sens = cohort_a_rumc_zgt_sensitivity(master)
    discordance_full, discordance_excl_outliers = discordance_sensitivity(master)

    cohort_b_3centre.to_csv(os.path.join(OUTPUT_DIR, "cohort_b_3centre_sensitivity.csv"), index=False)
    cohort_a_sens.to_csv(os.path.join(OUTPUT_DIR, "cohort_a_rumc_zgt_sensitivity.csv"), index=False)
    discordance_excl_outliers.to_csv(
        os.path.join(OUTPUT_DIR, "supp_discordance_sensitivity_excl_outliers.csv"), index=False)

    print_header("PHASE 1B TAMAMLANDI")
    print(f"Ciktilar kaydedildi: {OUTPUT_DIR}")
    print("  - cohort_b_3centre_sensitivity.csv")
    print("  - cohort_a_rumc_zgt_sensitivity.csv")
    print("  - supp_discordance_sensitivity_excl_outliers.csv")


if __name__ == "__main__":
    main()
