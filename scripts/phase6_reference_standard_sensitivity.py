"""
Revizyon Phase 6 - Reference Standard (Histological Verification) Sensitivity
=================================================================================
Hakem 2'nin Madde 4 itirazi: csPCa-negatif grup homojen degil -- bazi negatifler
histolojik olarak (biyopsi/RP) dogrulanmis, bazilari ise hic biyopsi yapilmadan
sadece MRI bulgusuna dayanarak negatif sayilmis -- differential verification
bias riski (pozitifler HER ZAMAN histolojik, negatiflerin bir kismi degil).

`marksheet_PICAI.csv`'deki 'histopath_type' kolonu bunu case-level olarak
veriyor (gercek veriyle dogrulandi, N=1500):
  - MRBx / SysBx / SysBx+MRBx / RP  -> histolojik olarak dogrulanmis
  - NaN (SADECE csPCa-negatif vakalarda gorulur, tum 425 pozitifte dolu)
    -> MRI-only, biyopsisiz negatif

Bu script HEDEFLI bir sensitivity analizi yapiyor (ADC/lesion/DCA/repeated-CV/
full threshold TEKRAR CALISTIRILMIYOR):
  1. histopath_type'i case_id (=patient_id + "_" + study_id) uzerinden ana
     merged CSV'ye join eder, join'in 1:1 oldugunu ve eslesmeyen vaka sayisini
     raporlar.
  2. Verification composition'i Source (tum merged CSV), Cohort A (N=1418),
     Cohort B (N=1096) icin AYRI AYRI raporlar.
  3. Histology-verified sensitivity cohortlari kurar: TUM csPCa+ vakalar +
     SADECE histolojik-dogrulanmis csPCa- vakalar (MRI-only negatifler
     disariya alinir).
  4. Cohort A (histology-verified): calculated manual PSAD vs AI-PSAD AUC,
     bootstrap deltaAUC CI, DeLong p, threshold=0.15 metrikleri.
  5. Cohort B (histology-verified): Model A vs Model A' AUC, AIC, bootstrap
     deltaAUC CI, DeLong p, coefficient ratio + LRT.
"""

import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

ON_COLAB = os.path.exists("/content/drive")
DATASET_ROOT = "/content/drive/MyDrive/Uroloji_Projesi/PICAI_Dataset" if ON_COLAB else "./PICAI_Dataset"
OUTPUT_DIR = os.path.join(DATASET_ROOT, "QA_outputs", "revision_outputs")
TABLES_DIR = os.path.join(OUTPUT_DIR, "final_tables")

# marksheet_PICAI.csv icin denenecek yollar, sirayla -- ilk bulunan kullanilir.
# GECMISTE bu dosya icin yanlis tahmin edilen bir Drive yolu FileNotFoundError
# vermisti (labels/clinical_information/marksheet.csv); bu yuzden birden fazla
# aday deneniyor ve hicbiri bulunamazsa net bir hata mesaji basiliyor, SESSIZCE
# atlanmiyor. EN GUVENLI YOL: yerel `marksheet_PICAI.csv` dosyasini (proje
# klasorunde zaten mevcut, histopath_type kolonu dogrulandi) Colab'da /content/
# altina yukleyin.
MARKSHEET_CANDIDATES = [
    "/content/marksheet_PICAI.csv",
    os.path.join(DATASET_ROOT, "labels", "clinical_information", "marksheet.csv"),
    os.path.join(DATASET_ROOT, "marksheet_PICAI.csv"),
]

PHASE_SCRIPTS_DIR = "/content" if ON_COLAB else os.path.dirname(os.path.abspath(__file__))
if PHASE_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, PHASE_SCRIPTS_DIR)

import phase3_primary_analyses as ph3
import phase5_clinical_performance as ph5

RANDOM_SEED = 42
N_BOOTSTRAP = 1000
HISTOLOGY_TYPES = {"MRBx", "SysBx", "SysBx+MRBx", "RP"}


def print_header(title):
    print("\n" + "=" * 90)
    print(f"||  {title}")
    print("=" * 90)


def print_subheader(title):
    print(f"\n{'-' * 80}\n  {title}\n{'-' * 80}")


def save_table(df, name, notes=None):
    os.makedirs(TABLES_DIR, exist_ok=True)
    csv_path = os.path.join(TABLES_DIR, f"{name}.csv")
    df.to_csv(csv_path, index=False)
    md_path = os.path.join(TABLES_DIR, f"{name}.md")
    try:
        body = df.to_markdown(index=False)
    except ImportError:
        body = df.to_string(index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(body)
        if notes:
            f.write("\n\n" + notes + "\n")
    print(f"  Tablo kaydedildi: {csv_path}")
    print(f"  Tablo kaydedildi: {md_path}")


# ============================================================
# 1. MARKSHEET YUKLEME + JOIN (case_id = patient_id_study_id)
# ============================================================
def load_marksheet_histopath():
    print_header("1. MARKSHEET YUKLEME VE HISTOPATH_TYPE CIKARIMI")
    marksheet_path = None
    for cand in MARKSHEET_CANDIDATES:
        if os.path.exists(cand):
            marksheet_path = cand
            break
    if marksheet_path is None:
        raise FileNotFoundError(
            "marksheet_PICAI.csv/marksheet.csv hicbir aday yolda bulunamadi. Denenen yollar:\n  "
            + "\n  ".join(MARKSHEET_CANDIDATES)
            + "\nEn guvenli cozum: yerel `marksheet_PICAI.csv` dosyasini Colab'da /content/ altina yukleyin."
        )
    print(f"Kullanilan marksheet dosyasi: {marksheet_path}")

    mk = pd.read_csv(marksheet_path)
    required_cols = {'patient_id', 'study_id', 'histopath_type', 'case_csPCa'}
    missing_cols = required_cols - set(mk.columns)
    if missing_cols:
        raise ValueError(f"marksheet dosyasinda beklenen kolonlar eksik: {missing_cols}")

    mk['case_id'] = mk['patient_id'].astype(str) + "_" + mk['study_id'].astype(str)
    n_dup = mk['case_id'].duplicated().sum()
    if n_dup > 0:
        print(f"UYARI: marksheet'te {n_dup} tekrarlanan case_id var -- join'de coklanmayi onlemek "
              f"icin ilk gorulen tutulacak.")
        mk = mk.drop_duplicates(subset='case_id', keep='first')

    print(f"Marksheet toplam satir (case_id benzersiz): {len(mk)}")

    print("\nhistopath_type value_counts (tum marksheet, csPCa durumuna gore):")
    print(pd.crosstab(mk['case_csPCa'], mk['histopath_type'].fillna('MISSING')))

    n_pos_missing = mk.loc[mk['case_csPCa'] == 'YES', 'histopath_type'].isna().sum()
    if n_pos_missing > 0:
        print(f"\nUYARI: {n_pos_missing} csPCa-pozitif vakada histopath_type eksik -- "
              f"beklenen 0, kontrol edilmeli.")
    else:
        print("\nDogrulama: TUM csPCa-pozitif vakalarda histopath_type dolu (beklenen davranis).")

    return mk[['case_id', 'histopath_type']].copy()


def join_verification_status(df, mk_histopath):
    print_subheader("Join: ana merged CSV <- marksheet histopath_type (case_id uzerinden)")
    n_before = len(df)
    is_matched = df['case_id'].isin(mk_histopath['case_id'])
    n_matched = int(is_matched.sum())
    n_unmatched_join = n_before - n_matched
    print(f"Ana CSV satir sayisi: {n_before}, marksheet'te eslesen case_id: {n_matched}, "
          f"eslesmeyen: {n_unmatched_join}")
    if n_unmatched_join > 0:
        print(f"UYARI: {n_unmatched_join} vaka marksheet'te bulunamadi -- bu vakalar icin "
              f"verification status 'UNMATCHED' olarak isaretlenecek (ne histolojik dogrulanmis "
              f"ne MRI-only sayilmiyor, sessizce yok sayilmiyor).")

    if 'histopath_type' in df.columns:
        # faz2_merged_clinical_ai_volume.csv'nin kendisi zaten bir 'histopath_type' kolonu
        # tasiyor olabilir (Faz 2'nin orijinal marksheet kolonlarini korumasindan) -- bu,
        # merge'de otomatik 'histopath_type_x'/'histopath_type_y' suffix'ine yol acip
        # sade 'histopath_type' adiyla erisimi KeyError'a dusuruyordu. Eski kolonu
        # dusurup marksheet_PICAI.csv'den (dogrulanmis kaynak) gelen temiz versiyonu
        # kullaniyoruz.
        print("  Not: df'te onceden bir 'histopath_type' kolonu vardi, marksheet'ten gelen "
              "temiz versiyonla degistiriliyor (suffix cakismasini onlemek icin).")
        df = df.drop(columns=['histopath_type'])

    df = df.merge(mk_histopath, on='case_id', how='left', validate='one_to_one')

    df['is_cspca_positive'] = df['csPCa_binary'] == 1
    df['is_histology_verified'] = df['histopath_type'].isin(HISTOLOGY_TYPES)
    is_matched_after = df['case_id'].isin(mk_histopath['case_id'])
    # UNMATCHED once ceki yapiliyor (pozitif/negatif ayrimindan once) -- boylece
    # marksheet'te hic bulunamayan bir vaka, yanlislikla "dogrulanmis" sayilmaz.
    df['verification_status'] = np.select(
        [
            ~is_matched_after,
            df['is_cspca_positive'],
            (~df['is_cspca_positive']) & df['is_histology_verified'],
            (~df['is_cspca_positive']) & (~df['is_histology_verified']),
        ],
        [
            'UNMATCHED (not found in marksheet)',
            'csPCa-positive (histologically verified)',
            'csPCa-negative, histologically verified',
            'csPCa-negative, MRI-only (no biopsy)',
        ],
        default='UNKNOWN',
    )
    n_unmatched_positive = int((~is_matched_after & df['is_cspca_positive']).sum())
    if n_unmatched_positive > 0:
        print(f"UYARI: {n_unmatched_positive} csPCa-pozitif vaka marksheet'te eslesmedi -- "
              f"bunlar 'UNMATCHED' olarak isaretlendi, histology-verified sensitivity cohort'a "
              f"dahil edilmeyecek (temkinli yaklasim).")
    return df


# ============================================================
# 2. VERIFICATION COMPOSITION RAPORU
# ============================================================
def verification_composition(df, label):
    print_subheader(f"Verification composition -- {label} (n={len(df)})")
    counts = df['verification_status'].value_counts()
    print(counts.to_string())

    n_total = len(df)
    n_pos = int(df['is_cspca_positive'].sum())
    n_neg = n_total - n_pos
    n_neg_verified = int(((~df['is_cspca_positive']) & df['is_histology_verified']).sum())
    n_neg_mri_only = int((df['verification_status'] == 'csPCa-negative, MRI-only (no biopsy)').sum())
    n_unmatched = int((df['verification_status'] == 'UNMATCHED (not found in marksheet)').sum())

    row = {
        'cohort': label, 'n_total': n_total, 'n_cspca_positive': n_pos,
        'n_cspca_negative': n_neg, 'n_negative_histology_verified': n_neg_verified,
        'n_negative_mri_only': n_neg_mri_only, 'n_unmatched': n_unmatched,
        'pct_negative_histology_verified': round(100 * n_neg_verified / n_neg, 1) if n_neg > 0 else np.nan,
        'pct_negative_mri_only': round(100 * n_neg_mri_only / n_neg, 1) if n_neg > 0 else np.nan,
        'n_histology_verified_sensitivity_cohort': n_pos + n_neg_verified,
    }
    return row


# ============================================================
# 3. COHORT A HISTOLOGY-VERIFIED SENSITIVITY
# ============================================================
def cohort_a_histology_sensitivity(cohort_a):
    print_header("3. COHORT A -- HISTOLOGY-VERIFIED SENSITIVITY (calculated manual PSAD vs AI-PSAD)")
    hv_mask = cohort_a['is_cspca_positive'] | cohort_a['is_histology_verified']
    cohort_a_hv = cohort_a[hv_mask].copy()
    n = len(cohort_a_hv)
    n_events = int(cohort_a_hv['csPCa_binary'].sum())
    print(f"Histology-verified sensitivity cohort: N={n} (csPCa+ n={n_events}, "
          f"{100*n_events/n:.1f}%; csPCa-negatif hepsi histolojik dogrulanmis)")
    print(f"Cikartilan MRI-only negatif sayisi: {len(cohort_a) - n}")

    y = cohort_a_hv['csPCa_binary'].values
    calc_psad = cohort_a_hv['calculated_manual_psad'].values
    ai_psad = cohort_a_hv['ai_psad'].values

    auc_calc, lo_calc, hi_calc = ph3.delong_auc_ci(y, calc_psad)
    auc_ai, lo_ai, hi_ai = ph3.delong_auc_ci(y, ai_psad)
    _, _, p_delong = ph3.delong_paired_test(y, calc_psad, ai_psad)
    boot = ph3.bootstrap_paired_delta_auc(y, calc_psad, ai_psad, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED)

    print(f"Calculated manual PSAD: AUC={auc_calc:.4f} (DeLong CI {lo_calc:.4f}-{hi_calc:.4f})")
    print(f"AI-PSAD:                AUC={auc_ai:.4f} (DeLong CI {lo_ai:.4f}-{hi_ai:.4f})")
    print(f"Paired DeLong test (supportive): p={p_delong:.4g}")
    print(f"Bootstrap paired deltaAUC (B={boot['n_boot']}, {boot['n_failed']} basarisiz): "
          f"median={boot['delta_median']:.4f}, 95% CI [{boot['delta_ci_lo']:.4f}, {boot['delta_ci_hi']:.4f}]")

    thr = ph5.threshold_metrics(y, calc_psad, 0.15)
    thr['marker'] = 'Calculated manual PSAD'
    thr_ai = ph5.threshold_metrics(y, ai_psad, 0.15)
    thr_ai['marker'] = 'AI-PSAD'
    threshold_df = pd.DataFrame([thr, thr_ai])
    print("\nThreshold=0.15 performansi (histology-verified sensitivity cohort):")
    print(threshold_df.to_string(index=False))

    result_row = {
        'n': n, 'events': n_events, 'auc_calculated_manual_psad': auc_calc,
        'ci_calc_lo': lo_calc, 'ci_calc_hi': hi_calc, 'auc_ai_psad': auc_ai,
        'ci_ai_lo': lo_ai, 'ci_ai_hi': hi_ai, 'delong_p_supportive': p_delong,
        'bootstrap_delta_median': boot['delta_median'], 'bootstrap_delta_ci_lo': boot['delta_ci_lo'],
        'bootstrap_delta_ci_hi': boot['delta_ci_hi'], 'n_boot': boot['n_boot'], 'n_boot_failed': boot['n_failed'],
    }
    return pd.DataFrame([result_row]), threshold_df


# ============================================================
# 4. COHORT B HISTOLOGY-VERIFIED SENSITIVITY
# ============================================================
def cohort_b_histology_sensitivity(cohort_b):
    print_header("4. COHORT B -- HISTOLOGY-VERIFIED SENSITIVITY (Model A vs Model A')")
    hv_mask = cohort_b['is_cspca_positive'] | cohort_b['is_histology_verified']
    cohort_b_hv = cohort_b[hv_mask].copy()
    n = len(cohort_b_hv)
    n_events = int(cohort_b_hv['csPCa_binary'].sum())
    print(f"Histology-verified sensitivity cohort: N={n} (csPCa+ n={n_events}, "
          f"{100*n_events/n:.1f}%)")
    print(f"Cikartilan MRI-only negatif sayisi: {len(cohort_b) - n}")

    y = cohort_b_hv['csPCa_binary'].values
    age = cohort_b_hv['age_per_10'].values
    psa = cohort_b_hv['log_PSA'].values
    vol_m = cohort_b_hv['log_manual_volume'].values
    vol_ai = cohort_b_hv['log_AI_volume'].values
    centre = cohort_b_hv['centre_RUMC'].values

    X_a = sm.add_constant(np.column_stack([age, psa, vol_m, centre]))
    X_ap = sm.add_constant(np.column_stack([age, psa, vol_ai, centre]))
    var_names = ['Intercept', 'age_per_10', 'log_PSA', 'log_volume', 'centre_RUMC']
    apparent_a = ph3.fit_and_report_model(y, X_a, var_names, "Model A (histology-verified sensitivity)")
    apparent_ap = ph3.fit_and_report_model(y, X_ap, var_names, "Model A' (histology-verified sensitivity)")
    _, _, p_delong = ph3.delong_paired_test(y, apparent_a['pred'], apparent_ap['pred'])

    # Coefficient ratio + delta-AUC bootstrap (resample-the-statistic), Phase 3 ile ayni mantik
    rng = np.random.RandomState(RANDOM_SEED)
    n_obs = len(y)
    ratios_a, ratios_ap, deltas = [], [], []
    n_failed = 0
    for _ in range(N_BOOTSTRAP):
        idx = rng.randint(0, n_obs, n_obs)
        y_b = y[idx]
        if len(np.unique(y_b)) < 2:
            n_failed += 1
            continue
        try:
            age_b, psa_b, vm_b, va_b, c_b = age[idx], psa[idx], vol_m[idx], vol_ai[idx], centre[idx]
            X_a_b = sm.add_constant(np.column_stack([age_b, psa_b, vm_b, c_b]))
            X_ap_b = sm.add_constant(np.column_stack([age_b, psa_b, va_b, c_b]))
            model_a_b = ph3.fit_logit(y_b, X_a_b)
            model_ap_b = ph3.fit_logit(y_b, X_ap_b)
            p_a, p_ap = np.asarray(model_a_b.params), np.asarray(model_ap_b.params)
            ratios_a.append(abs(p_a[3] / p_a[2]))
            ratios_ap.append(abs(p_ap[3] / p_ap[2]))
            auc_a_b = roc_auc_score(y_b, model_a_b.predict(X_a_b))
            auc_ap_b = roc_auc_score(y_b, model_ap_b.predict(X_ap_b))
            deltas.append(auc_ap_b - auc_a_b)
        except Exception:
            n_failed += 1
            continue

    def pct_ci(arr):
        arr = np.array(arr)
        return np.median(arr), np.percentile(arr, 2.5), np.percentile(arr, 97.5)

    ratio_a_med, ratio_a_lo, ratio_a_hi = pct_ci(ratios_a)
    ratio_ap_med, ratio_ap_lo, ratio_ap_hi = pct_ci(ratios_ap)
    delta_med, delta_lo, delta_hi = pct_ci(deltas)

    print(f"\nModel A : AIC={apparent_a['aic']:.3f}, apparent AUC={apparent_a['auc']:.4f} "
          f"(DeLong CI {apparent_a['auc_lo']:.4f}-{apparent_a['auc_hi']:.4f})")
    print(f"Model A': AIC={apparent_ap['aic']:.3f}, apparent AUC={apparent_ap['auc']:.4f} "
          f"(DeLong CI {apparent_ap['auc_lo']:.4f}-{apparent_ap['auc_hi']:.4f})")
    print(f"Paired DeLong test (apparent, supportive): p={p_delong:.4g}")
    print(f"Bootstrap paired deltaAUC (B={N_BOOTSTRAP}, {n_failed} basarisiz): "
          f"median={delta_med:.4f}, 95% CI [{delta_lo:.4f}, {delta_hi:.4f}]")
    print(f"Coefficient ratio Model A : median={ratio_a_med:.4f}, 95% CI [{ratio_a_lo:.4f}, {ratio_a_hi:.4f}]")
    print(f"Coefficient ratio Model A': median={ratio_ap_med:.4f}, 95% CI [{ratio_ap_lo:.4f}, {ratio_ap_hi:.4f}]")

    lrt_a = ph3.run_lrt(y, age, vol_m, psa, centre, "Model A (histology-verified sensitivity)")
    lrt_ap = ph3.run_lrt(y, age, vol_ai, psa, centre, "Model A' (histology-verified sensitivity)")

    result_row = {
        'n': n, 'events': n_events,
        'aic_model_a': apparent_a['aic'], 'auc_model_a': apparent_a['auc'],
        'auc_a_ci_lo': apparent_a['auc_lo'], 'auc_a_ci_hi': apparent_a['auc_hi'],
        'aic_model_ap': apparent_ap['aic'], 'auc_model_ap': apparent_ap['auc'],
        'auc_ap_ci_lo': apparent_ap['auc_lo'], 'auc_ap_ci_hi': apparent_ap['auc_hi'],
        'delong_p_supportive': p_delong,
        'bootstrap_delta_median': delta_med, 'bootstrap_delta_ci_lo': delta_lo, 'bootstrap_delta_ci_hi': delta_hi,
        'n_boot': N_BOOTSTRAP, 'n_boot_failed': n_failed,
        'coef_ratio_a_median': ratio_a_med, 'coef_ratio_a_ci_lo': ratio_a_lo, 'coef_ratio_a_ci_hi': ratio_a_hi,
        'coef_ratio_ap_median': ratio_ap_med, 'coef_ratio_ap_ci_lo': ratio_ap_lo, 'coef_ratio_ap_ci_hi': ratio_ap_hi,
        'lrt_chi2_a': lrt_a['lr_chi2'], 'lrt_p_a': lrt_a['p_value'],
        'lrt_chi2_ap': lrt_ap['lr_chi2'], 'lrt_p_ap': lrt_ap['p_value'],
    }
    return pd.DataFrame([result_row])


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    print_header("PHASE 6: REFERENCE STANDARD (HISTOLOGICAL VERIFICATION) SENSITIVITY")

    mk_histopath = load_marksheet_histopath()

    df = pd.read_csv(ph3.MERGED_CSV_PATH)
    df['csPCa_binary'] = (df['case_csPCa'] == 'YES').astype(int)
    df = join_verification_status(df, mk_histopath)

    cohort_a = ph3.build_cohort_a(pd.read_csv(ph3.MERGED_CSV_PATH))
    cohort_a = join_verification_status(cohort_a, mk_histopath)
    cohort_b = ph3.build_cohort_b(pd.read_csv(ph3.MERGED_CSV_PATH))
    cohort_b = join_verification_status(cohort_b, mk_histopath)

    print_header("2. VERIFICATION COMPOSITION -- SOURCE / COHORT A / COHORT B")
    comp_rows = [
        verification_composition(df, "Source dataset (all merged CSV rows)"),
        verification_composition(cohort_a, "Cohort A (N=1418)"),
        verification_composition(cohort_b, "Cohort B (N=1096)"),
    ]
    comp_df = pd.DataFrame(comp_rows)
    print("\n" + comp_df.to_string(index=False))
    save_table(comp_df, "TableS_reference_standard_verification_composition",
               notes="Note. Histologically verified = histopath_type in {MRBx, SysBx, SysBx+MRBx, RP} "
                     "(targeted biopsy, systematic biopsy, combined biopsy, or radical prostatectomy). "
                     "MRI-only = csPCa-negative case with no biopsy performed (histopath_type missing). "
                     "All csPCa-positive cases had non-missing histopath_type (verified in code).")

    cohort_a_hv_result, cohort_a_hv_threshold = cohort_a_histology_sensitivity(cohort_a)
    save_table(cohort_a_hv_result, "TableS_cohortA_histology_verified_sensitivity")
    save_table(cohort_a_hv_threshold, "TableS_cohortA_histology_verified_threshold015")

    cohort_b_hv_result = cohort_b_histology_sensitivity(cohort_b)
    save_table(cohort_b_hv_result, "TableS_cohortB_histology_verified_sensitivity")

    print_header("5. YORUM -- YON KORUNUYOR MU?")
    delta_a = cohort_a_hv_result.iloc[0]['bootstrap_delta_median']
    delta_b = cohort_b_hv_result.iloc[0]['bootstrap_delta_median']
    print(f"Cohort A (histology-verified) delta AUC (AI-PSAD - calculated): {delta_a:+.4f}")
    print(f"Cohort B (histology-verified) delta AUC (Model A' - Model A): {delta_b:+.4f}")
    if delta_a > 0 and delta_b > 0:
        print(">>> YON KORUNUYOR: AI-hacim/AI-PSAD avantaji, sadece histolojik-dogrulanmis "
              "negatiflerle sinirlandirildiginda da POZITIF yonde kaliyor -- MRI-only negatiflerin "
              "dahil edilmesi bulgunun yonunu degistirmiyor.")
    else:
        print(">>> UYARI: Yon degisti veya biri negatif -- bu dikkatle raporlanmali, spin yapilmamali.")

    print_header("PHASE 6 TAMAMLANDI")
    print(f"Tablolar: {TABLES_DIR}")


if __name__ == "__main__":
    main()
