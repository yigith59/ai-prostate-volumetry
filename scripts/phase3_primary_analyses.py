"""
Revizyon Phase 3 - Primer Analizler
=================================================================================
Phase 2'nin sonucu ("Log-linear form acceptable") bu fazin onkosuluydu --
artik katsayi orani ve LRT hicbir cekince eklenmeden raporlanabilir.

Bu script alti alt-analiz yapiyor:
3.1  Model A / Model A' yeniden fit (Cohort B, N=1096)
3.2  LRT -- PSAD esit-agirlik kisiti testi (her iki model icin)
3.3  Primer single-marker karsilastirma: calculated manual PSAD vs AI-PSAD
     (Cohort A, N=1418) + reported PSAD secondary (N=1033 alt-kume)
3.4  Bootstrap coefficient-ratio CI (Cohort B)
3.5  Optimism-corrected kalibrasyon (Cohort B)
3.6  Repeated k-fold duzeltmesi (pooled OOF, "empirical range", "95% CI" DEGIL)

VERIMLILIK KARARI: 3.4 ve 3.5 AYNI bootstrap dongusunde (B=1000) birlikte
hesaplaniyor -- her resample'da Model A/A' zaten yeniden fit ediliyor, bu
fit'ten hem katsayi orani hem apparent/test AUC-Brier-kalibrasyon cikarilir.
2000 yerine 1000 model fit'i yeterli.

Kohortlar Phase 1/2 ile BIREBIR AYNI kriterle bu script icinde yeniden
kuruluyor (self-contained convention).
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

DATASET_ROOT = "/content/drive/MyDrive/Uroloji_Projesi/PICAI_Dataset"
MERGED_CSV_PATH = os.path.join(DATASET_ROOT, "QA_outputs", "faz2_merged_clinical_ai_volume.csv")
OUTPUT_DIR = os.path.join(DATASET_ROOT, "QA_outputs", "revision_outputs")

RANDOM_SEED = 42
N_BOOTSTRAP = 1000
N_KFOLD_REPEATS = 20
K_FOLDS = 5

EXPECTED_N_COHORT_A = 1418
EXPECTED_N_COHORT_B = 1096


def print_header(title):
    print("\n" + "=" * 90)
    print(f"||  {title}")
    print("=" * 90)


def print_subheader(title):
    print(f"\n{'-' * 80}\n  {title}\n{'-' * 80}")


# ============================================================
# KOHORTLAR (Phase 1/2 ile birebir ayni kriter)
# ============================================================
def load_raw():
    return pd.read_csv(MERGED_CSV_PATH)


def build_cohort_a(df):
    mask = (
        df['psa'].notna() &
        df['prostate_volume'].notna() &
        df['ai_volume_Bosma22b'].notna() &
        df['case_csPCa'].notna()
    )
    cohort = df[mask].copy()
    cohort['csPCa_binary'] = (cohort['case_csPCa'] == 'YES').astype(int)
    cohort['calculated_manual_psad'] = cohort['psa'] / cohort['prostate_volume']
    cohort['ai_psad'] = cohort['psa'] / cohort['ai_volume_Bosma22b']
    cohort['reported_psad'] = cohort['psad']  # CSV'deki ham kolon adi 'psad' -- acik isimle kopyalaniyor
    n = len(cohort)
    if n != EXPECTED_N_COHORT_A:
        print(f"UYARI: Cohort A beklenen N={EXPECTED_N_COHORT_A} degil, N={n}.")
    return cohort


def build_cohort_b(df):
    mask = (
        df['patient_age'].notna() &
        df['psa'].notna() &
        df['prostate_volume'].notna() &
        df['ai_volume_Bosma22b'].notna() &
        df['center'].notna() &
        df['case_csPCa'].notna() &
        (df['center'] != 'PCNN')
    )
    cohort = df[mask].copy()
    cohort['csPCa_binary'] = (cohort['case_csPCa'] == 'YES').astype(int)
    cohort['age_per_10'] = cohort['patient_age'] / 10.0
    # DUZELTME: log1p degil gercek log(x) -- Phase 2 script'indeki ayni gerekce gecerli:
    # conventional PSAD constraint log(PSA/hacim)=log(PSA)-log(hacim) sadece gercek log ile
    # matematiksel olarak tam esdeger. Sifir/negatif kontrolu yapiliyor, hiçbiri varsa hata verir.
    n_psa_nonpos = (cohort['psa'] <= 0).sum()
    n_manual_nonpos = (cohort['prostate_volume'] <= 0).sum()
    n_ai_nonpos = (cohort['ai_volume_Bosma22b'] <= 0).sum()
    print(f"Sifir/negatif deger kontrolu -- PSA<=0: {n_psa_nonpos}, manuel hacim<=0: {n_manual_nonpos}, "
          f"AI hacim<=0: {n_ai_nonpos}")
    if n_psa_nonpos > 0 or n_manual_nonpos > 0 or n_ai_nonpos > 0:
        raise ValueError("PSA/hacimde sifir veya negatif deger bulundu -- gercek log() kullanilamaz, "
                          "bu vakalar once ozel olarak ele alinmali.")
    cohort['log_PSA'] = np.log(cohort['psa'])
    cohort['log_manual_volume'] = np.log(cohort['prostate_volume'])
    cohort['log_AI_volume'] = np.log(cohort['ai_volume_Bosma22b'])
    cohort['centre_RUMC'] = (cohort['center'] == 'RUMC').astype(int)
    n = len(cohort)
    if n != EXPECTED_N_COHORT_B:
        print(f"UYARI: Cohort B beklenen N={EXPECTED_N_COHORT_B} degil, N={n}.")
    return cohort


# ============================================================
# LOJISTIK REGRESYON / DELONG YARDIMCILARI
# ============================================================
def fit_logit(y, X):
    model = sm.Logit(y, X).fit(disp=0)
    if not model.mle_retvals.get('converged', True):
        raise RuntimeError("Model yakinsamadi.")
    return model


def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted_T, m):
    n = preds_sorted_T.shape[1] - m
    pos = preds_sorted_T[:, :m]
    neg = preds_sorted_T[:, m:]
    k = preds_sorted_T.shape[0]
    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(pos[r, :])
        ty[r, :] = _compute_midrank(neg[r, :])
        tz[r, :] = _compute_midrank(preds_sorted_T[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    # np.cov tek satirli (k=1) girdide SKALER donduruyor, 1x1 matris degil --
    # bu, tek-marker DeLong CI cagrisinda (delong_auc_ci, k=1) cov[0,0]
    # indekslemesini patlatir. atleast_2d hem k=1 (skaler->1x1) hem k=2
    # (zaten 2x2, degismez) durumunu güvenli hale getirir.
    delongcov = np.atleast_2d(sx / m + sy / n)
    return aucs, delongcov


def delong_auc_ci(y, pred, ci=0.95):
    """Tek bir marker/model icin DeLong AUC + Normal-teori CI."""
    y = np.asarray(y)
    order = np.argsort(-y)
    y_sorted = y[order]
    m = int(np.sum(y_sorted))
    preds = np.asarray(pred)[order].reshape(1, -1)
    aucs, cov = _fast_delong(preds, m)
    se = np.sqrt(cov[0, 0])
    z = stats.norm.ppf(1 - (1 - ci) / 2)
    return aucs[0], aucs[0] - z * se, aucs[0] + z * se


def delong_paired_test(y, pred1, pred2):
    y = np.asarray(y)
    order = np.argsort(-y)
    y_sorted = y[order]
    m = int(np.sum(y_sorted))
    preds = np.vstack([np.asarray(pred1), np.asarray(pred2)])[:, order]
    aucs, cov = _fast_delong(preds, m)
    l = np.array([[1, -1]])
    var = float(np.dot(np.dot(l, cov), l.T))
    z = abs(aucs[0] - aucs[1]) / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(z))
    return aucs[0], aucs[1], p


def brier_score(y, pred):
    return float(np.mean((np.asarray(y) - np.asarray(pred)) ** 2))


def calibration_slope_intercept(y, pred, eps=1e-6):
    pred_clip = np.clip(pred, eps, 1 - eps)
    logit_pred = np.log(pred_clip / (1 - pred_clip))
    X = sm.add_constant(logit_pred)
    m = sm.Logit(y, X).fit(disp=0)
    p = np.asarray(m.params)
    return p[0], p[1]  # intercept, slope


def mcfadden_pseudo_r2(model, y):
    p_bar = np.mean(y)
    ll_null = np.sum(y * np.log(p_bar) + (1 - y) * np.log(1 - p_bar))
    return 1 - model.llf / ll_null


# ============================================================
# 3.1 MODEL A / A' YENIDEN FIT
# ============================================================
def fit_and_report_model(y, X, var_names, label):
    print_subheader(label)
    model = fit_logit(y, X)
    pred = model.predict(X)
    auc, auc_lo, auc_hi = delong_auc_ci(y, pred)
    pseudo_r2 = mcfadden_pseudo_r2(model, y)

    params = np.asarray(model.params)
    ses = np.asarray(model.bse)
    pvals = np.asarray(model.pvalues)

    print(f"{'Variable':<22} | {'Coef':>10} | {'SE':>8} | {'OR':>8} | {'95% CI':>18} | {'p':>10}")
    for i, name in enumerate(var_names):
        coef, se, p = params[i], ses[i], pvals[i]
        if name == 'Intercept':
            print(f"{name:<22} | {coef:>10.4f} | {se:>8.4f} | {'-':>8} | {'-':>18} | {p:>10.4g}")
        else:
            or_val = np.exp(coef)
            ci_lo, ci_hi = np.exp(coef - 1.96 * se), np.exp(coef + 1.96 * se)
            print(f"{name:<22} | {coef:>10.4f} | {se:>8.4f} | {or_val:>8.3f} | "
                  f"({ci_lo:.3f}-{ci_hi:.3f}) | {p:>10.4g}")
    print(f"\nAIC={model.aic:.3f}, pseudo-R2(McFadden)={pseudo_r2:.4f}")
    print(f"Apparent AUC={auc:.4f} (DeLong 95% CI {auc_lo:.4f}-{auc_hi:.4f})")

    return {
        'label': label, 'n': len(y), 'events': int(np.sum(y)),
        'params': params, 'ses': ses, 'pvals': pvals, 'var_names': var_names,
        'aic': model.aic, 'pseudo_r2': pseudo_r2,
        'auc': auc, 'auc_lo': auc_lo, 'auc_hi': auc_hi,
        'model': model, 'pred': pred,
    }


# ============================================================
# 3.2 LRT -- PSAD ESIT-AGIRLIK KISITI
# ============================================================
def run_lrt(y, age, target_log_vol, psa_log, centre, label):
    print_subheader(f"LRT (esit-agirlik kisiti) -- {label}")
    X_free = sm.add_constant(np.column_stack([age, psa_log, target_log_vol, centre]))
    model_free = fit_logit(y, X_free)

    composite = psa_log - target_log_vol
    X_restr = sm.add_constant(np.column_stack([age, composite, centre]))
    model_restr = fit_logit(y, X_restr)

    df_diff = X_free.shape[1] - X_restr.shape[1]
    lr_chi2 = 2 * (model_free.llf - model_restr.llf)
    p_value = chi2.sf(lr_chi2, df=df_diff)
    composite_coef = np.asarray(model_restr.params)[2]

    print(f"Serbest model log-lik={model_free.llf:.3f} (AIC={model_free.aic:.1f})")
    print(f"Kisitli model (log(PSA/hacim) composite) log-lik={model_restr.llf:.3f} (AIC={model_restr.aic:.1f})")
    print(f"Kisitli modelde composite katsayi={composite_coef:.4f}")
    print(f"LRT: chi2={lr_chi2:.3f}, df={df_diff}, p={p_value:.4g}")

    return {'label': label, 'lr_chi2': lr_chi2, 'df': df_diff, 'p_value': p_value,
            'composite_coef': composite_coef, 'll_free': model_free.llf, 'll_restricted': model_restr.llf,
            'aic_free': model_free.aic, 'aic_restricted': model_restr.aic}


# ============================================================
# 3.3 PRIMER SINGLE-MARKER KARSILASTIRMA (Cohort A) + REPORTED PSAD SECONDARY
# ============================================================
def bootstrap_paired_delta_auc(y, pred1, pred2, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    pred1 = np.asarray(pred1)
    pred2 = np.asarray(pred2)
    n = len(y)
    deltas = []
    n_failed = 0
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2:
            n_failed += 1
            continue
        try:
            auc1 = roc_auc_score(y[idx], pred1[idx])
            auc2 = roc_auc_score(y[idx], pred2[idx])
            deltas.append(auc2 - auc1)
        except Exception:
            n_failed += 1
    deltas = np.array(deltas)
    return {
        'n_boot': n_boot, 'n_failed': n_failed,
        'delta_median': np.median(deltas), 'delta_ci_lo': np.percentile(deltas, 2.5),
        'delta_ci_hi': np.percentile(deltas, 97.5),
    }


def run_primary_single_marker(cohort_a):
    print_header("3.3 PRIMER SINGLE-MARKER KARSILASTIRMA (Cohort A, N=%d)" % len(cohort_a))
    y = cohort_a['csPCa_binary'].values
    calc_psad = cohort_a['calculated_manual_psad'].values
    ai_psad = cohort_a['ai_psad'].values

    auc_calc, lo_calc, hi_calc = delong_auc_ci(y, calc_psad)
    auc_ai, lo_ai, hi_ai = delong_auc_ci(y, ai_psad)
    auc1, auc2, p_delong = delong_paired_test(y, calc_psad, ai_psad)

    print(f"Calculated manual PSAD: AUC={auc_calc:.4f} (DeLong CI {lo_calc:.4f}-{hi_calc:.4f})")
    print(f"AI-PSAD:                AUC={auc_ai:.4f} (DeLong CI {lo_ai:.4f}-{hi_ai:.4f})")
    print(f"Paired DeLong test: p={p_delong:.4g}")

    boot = bootstrap_paired_delta_auc(y, calc_psad, ai_psad)
    print(f"Bootstrap paired ΔAUC (B={boot['n_boot']}, {boot['n_failed']} basarisiz): "
          f"median={boot['delta_median']:.4f}, 95% CI [{boot['delta_ci_lo']:.4f}, {boot['delta_ci_hi']:.4f}]")

    # Secondary: reported PSAD (N=1033 alt-kume, ayri satir, Cohort A'nin PARCASI DEGIL).
    # KRITIK: reported PSAD'i calculated-manual-PSAD/AI-PSAD ile ADIL kiyaslamak icin
    # ikisi de AYNI N=1033 alt-kumesinde de hesaplanmali -- aksi halde N=1418 (PCNN dahil,
    # daha genis/heterojen) vs N=1033 (PCNN'siz) farki, "reported PSAD daha iyi" gibi
    # yanlis bir izlenim yaratabilir (populasyon farki, veri kalitesi farki degil).
    sub = cohort_a.dropna(subset=['reported_psad'])
    reported_row = None
    if len(sub) > 0:
        y_sub = sub['csPCa_binary'].values
        auc_rep, lo_rep, hi_rep = delong_auc_ci(y_sub, sub['reported_psad'].values)
        auc_calc_sub, lo_calc_sub, hi_calc_sub = delong_auc_ci(y_sub, sub['calculated_manual_psad'].values)
        auc_ai_sub, lo_ai_sub, hi_ai_sub = delong_auc_ci(y_sub, sub['ai_psad'].values)
        print(f"\n[SECONDARY, N={len(sub)} -- AYNI ALT-KUMEDE UC MARKER DE HESAPLANDI, ADIL KIYAS]")
        print(f"  Reported PSAD:          AUC={auc_rep:.4f} (DeLong CI {lo_rep:.4f}-{hi_rep:.4f})")
        print(f"  Calculated manual PSAD: AUC={auc_calc_sub:.4f} (DeLong CI {lo_calc_sub:.4f}-{hi_calc_sub:.4f})")
        print(f"  AI-PSAD:                AUC={auc_ai_sub:.4f} (DeLong CI {lo_ai_sub:.4f}-{hi_ai_sub:.4f})")
        reported_row = {'n': len(sub), 'auc_reported': auc_rep, 'auc_reported_ci': (lo_rep, hi_rep),
                         'auc_calculated_same_subset': auc_calc_sub, 'auc_ai_same_subset': auc_ai_sub}

    return {
        'n': len(cohort_a), 'events': int(y.sum()),
        'auc_calculated_manual_psad': auc_calc, 'ci_calc': (lo_calc, hi_calc),
        'auc_ai_psad': auc_ai, 'ci_ai': (lo_ai, hi_ai),
        'delong_p': p_delong,
        'bootstrap_delta': boot,
        'reported_psad_secondary': reported_row,
    }


# ============================================================
# 3.4 + 3.5: BIRLESIK BOOTSTRAP -- COEFFICIENT RATIO CI + OPTIMISM-CORRECTED CALIBRATION
# ============================================================
def combined_bootstrap_ratio_and_calibration(cohort_b, apparent_a, apparent_ap, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED):
    print_header(f"3.4+3.5 BIRLESIK BOOTSTRAP (B={n_boot}) -- coefficient ratio CI + optimism-corrected calibration")
    rng = np.random.RandomState(seed)
    n = len(cohort_b)
    y_full = cohort_b['csPCa_binary'].values
    age_full = cohort_b['age_per_10'].values
    psa_full = cohort_b['log_PSA'].values
    vol_manual_full = cohort_b['log_manual_volume'].values
    vol_ai_full = cohort_b['log_AI_volume'].values
    centre_full = cohort_b['centre_RUMC'].values

    ratios_a, ratios_ap = [], []
    apparent_auc_a, test_auc_a, apparent_auc_ap, test_auc_ap = [], [], [], []
    apparent_brier_a, test_brier_a, apparent_brier_ap, test_brier_ap = [], [], [], []
    apparent_calib_a, test_calib_a, apparent_calib_ap, test_calib_ap = [], [], [], []
    delta_auc_test = []
    delta_auc_apparent = []
    n_failed = 0

    # Orijinal (tam) kohort tasarim matrisleri -- her resample'da AYNI, dongu icinde
    # tekrar tekrar kurmak yerine BIR KEZ hesaplaniyor (1000x gereksiz islem onlendi).
    X_a_full = sm.add_constant(np.column_stack([age_full, psa_full, vol_manual_full, centre_full]))
    X_ap_full = sm.add_constant(np.column_stack([age_full, psa_full, vol_ai_full, centre_full]))

    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        y_b = y_full[idx]
        if len(np.unique(y_b)) < 2:
            n_failed += 1
            continue
        try:
            age_b, psa_b, vm_b, va_b, c_b = age_full[idx], psa_full[idx], vol_manual_full[idx], vol_ai_full[idx], centre_full[idx]

            X_a_b = sm.add_constant(np.column_stack([age_b, psa_b, vm_b, c_b]))
            model_a_b = fit_logit(y_b, X_a_b)
            X_ap_b = sm.add_constant(np.column_stack([age_b, psa_b, va_b, c_b]))
            model_ap_b = fit_logit(y_b, X_ap_b)

            p_a = np.asarray(model_a_b.params)
            p_ap = np.asarray(model_ap_b.params)
            ratios_a.append(abs(p_a[3] / p_a[2]))
            ratios_ap.append(abs(p_ap[3] / p_ap[2]))

            pred_apparent_a = model_a_b.predict(X_a_b)
            pred_apparent_ap = model_ap_b.predict(X_ap_b)

            pred_test_a = model_a_b.predict(X_a_full)
            pred_test_ap = model_ap_b.predict(X_ap_full)

            auc_apparent_a_b = roc_auc_score(y_b, pred_apparent_a)
            auc_test_a_b = roc_auc_score(y_full, pred_test_a)
            auc_apparent_ap_b = roc_auc_score(y_b, pred_apparent_ap)
            auc_test_ap_b = roc_auc_score(y_full, pred_test_ap)
            apparent_auc_a.append(auc_apparent_a_b)
            test_auc_a.append(auc_test_a_b)
            apparent_auc_ap.append(auc_apparent_ap_b)
            test_auc_ap.append(auc_test_ap_b)
            delta_auc_test.append(auc_test_ap_b - auc_test_a_b)
            # PRIMARY delta-AUC CI icin: resample'da fit edilen model, AYNI resample'da
            # degerlendiriliyor (standart "resample the statistic" bootstrap) -- bu,
            # yukaridaki delta_auc_test'ten (sabit orijinal kohortta degerlendirme, sadece
            # optimism-tahmini icin uygun, gercek orneklem belirsizligini eksik yakalar)
            # FARKLI ve daha genis/dogru bir belirsizlik olcusu verir.
            delta_auc_apparent.append(auc_apparent_ap_b - auc_apparent_a_b)

            apparent_brier_a.append(brier_score(y_b, pred_apparent_a))
            test_brier_a.append(brier_score(y_full, pred_test_a))
            apparent_brier_ap.append(brier_score(y_b, pred_apparent_ap))
            test_brier_ap.append(brier_score(y_full, pred_test_ap))

            apparent_calib_a.append(calibration_slope_intercept(y_b, pred_apparent_a))
            test_calib_a.append(calibration_slope_intercept(y_full, pred_test_a))
            apparent_calib_ap.append(calibration_slope_intercept(y_b, pred_apparent_ap))
            test_calib_ap.append(calibration_slope_intercept(y_full, pred_test_ap))
        except Exception:
            n_failed += 1
            continue

    print(f"Basarili resample: {n_boot - n_failed}/{n_boot} (basarisiz: {n_failed})")

    def pct_ci(arr):
        return np.median(arr), np.percentile(arr, 2.5), np.percentile(arr, 97.5)

    ratio_a_med, ratio_a_lo, ratio_a_hi = pct_ci(ratios_a)
    ratio_ap_med, ratio_ap_lo, ratio_ap_hi = pct_ci(ratios_ap)

    point_est_a = abs(apparent_a['params'][3] / apparent_a['params'][2])
    point_est_ap = abs(apparent_ap['params'][3] / apparent_ap['params'][2])
    print(f"\n[3.4] Coefficient ratio (|beta_vol/beta_PSA|):")
    print(f"  Model A : point-estimate={point_est_a:.4f}, bootstrap median={ratio_a_med:.4f}, "
          f"95% CI [{ratio_a_lo:.4f}, {ratio_a_hi:.4f}]")
    print(f"  Model A': point-estimate={point_est_ap:.4f}, bootstrap median={ratio_ap_med:.4f}, "
          f"95% CI [{ratio_ap_lo:.4f}, {ratio_ap_hi:.4f}]")

    optimism_auc_a = np.mean(np.array(apparent_auc_a) - np.array(test_auc_a))
    optimism_auc_ap = np.mean(np.array(apparent_auc_ap) - np.array(test_auc_ap))
    corrected_auc_a = apparent_a['auc'] - optimism_auc_a
    corrected_auc_ap = apparent_ap['auc'] - optimism_auc_ap

    optimism_brier_a = np.mean(np.array(apparent_brier_a) - np.array(test_brier_a))
    optimism_brier_ap = np.mean(np.array(apparent_brier_ap) - np.array(test_brier_ap))
    apparent_brier_a_full = brier_score(y_full, apparent_a['pred'])
    apparent_brier_ap_full = brier_score(y_full, apparent_ap['pred'])
    corrected_brier_a = apparent_brier_a_full - optimism_brier_a
    corrected_brier_ap = apparent_brier_ap_full - optimism_brier_ap

    apparent_calib_a_full = calibration_slope_intercept(y_full, apparent_a['pred'])
    apparent_calib_ap_full = calibration_slope_intercept(y_full, apparent_ap['pred'])
    opt_int_a = np.mean([a[0] - t[0] for a, t in zip(apparent_calib_a, test_calib_a)])
    opt_slope_a = np.mean([a[1] - t[1] for a, t in zip(apparent_calib_a, test_calib_a)])
    opt_int_ap = np.mean([a[0] - t[0] for a, t in zip(apparent_calib_ap, test_calib_ap)])
    opt_slope_ap = np.mean([a[1] - t[1] for a, t in zip(apparent_calib_ap, test_calib_ap)])
    corrected_int_a = apparent_calib_a_full[0] - opt_int_a
    corrected_slope_a = apparent_calib_a_full[1] - opt_slope_a
    corrected_int_ap = apparent_calib_ap_full[0] - opt_int_ap
    corrected_slope_ap = apparent_calib_ap_full[1] - opt_slope_ap

    print(f"\n[3.5] Optimism-corrected performans:")
    print(f"  Model A : apparent AUC={apparent_a['auc']:.4f}, optimism={optimism_auc_a:.4f}, "
          f"corrected AUC={corrected_auc_a:.4f}")
    print(f"            apparent Brier={apparent_brier_a_full:.4f}, corrected Brier={corrected_brier_a:.4f}")
    print(f"            apparent calib intercept={apparent_calib_a_full[0]:.4f}, "
          f"corrected intercept={corrected_int_a:.4f}")
    print(f"            apparent calib slope={apparent_calib_a_full[1]:.4f}, "
          f"corrected slope={corrected_slope_a:.4f}")
    print(f"  Model A': apparent AUC={apparent_ap['auc']:.4f}, optimism={optimism_auc_ap:.4f}, "
          f"corrected AUC={corrected_auc_ap:.4f}")
    print(f"            apparent Brier={apparent_brier_ap_full:.4f}, corrected Brier={corrected_brier_ap:.4f}")
    print(f"            apparent calib intercept={apparent_calib_ap_full[0]:.4f}, "
          f"corrected intercept={corrected_int_ap:.4f}")
    print(f"            apparent calib slope={apparent_calib_ap_full[1]:.4f}, "
          f"corrected slope={corrected_slope_ap:.4f}")

    delta_med_apparent, delta_lo_apparent, delta_hi_apparent = pct_ci(delta_auc_apparent)
    print(f"\n[PRIMARY] Bootstrap ΔAUC (A'-A), standart resample-the-statistic yontemi "
          f"(her resample kendi uzerinde degerlendiriliyor): median={delta_med_apparent:.4f}, "
          f"95% CI [{delta_lo_apparent:.4f}, {delta_hi_apparent:.4f}]")

    delta_med_test, delta_lo_test, delta_hi_test = pct_ci(delta_auc_test)
    print(f"[DIAGNOSTIC, optimism-icin] Bootstrap ΔAUC, sabit orijinal kohortta test edilerek: "
          f"median={delta_med_test:.4f}, 95% CI [{delta_lo_test:.4f}, {delta_hi_test:.4f}] "
          f"(bu, orneklem belirsizligini eksik yakalar -- manuscript'te PRIMARY olarak KULLANMA)")

    return {
        'n_boot': n_boot, 'n_failed': n_failed,
        'ratio_a': {'median': ratio_a_med, 'ci_lo': ratio_a_lo, 'ci_hi': ratio_a_hi,
                    'point_estimate': abs(apparent_a['params'][3] / apparent_a['params'][2])},
        'ratio_ap': {'median': ratio_ap_med, 'ci_lo': ratio_ap_lo, 'ci_hi': ratio_ap_hi,
                     'point_estimate': abs(apparent_ap['params'][3] / apparent_ap['params'][2])},
        'model_a_calibration': {'apparent_auc': apparent_a['auc'], 'optimism_auc': optimism_auc_a,
                                 'corrected_auc': corrected_auc_a, 'apparent_brier': apparent_brier_a_full,
                                 'corrected_brier': corrected_brier_a,
                                 'apparent_intercept': apparent_calib_a_full[0], 'corrected_intercept': corrected_int_a,
                                 'apparent_slope': apparent_calib_a_full[1], 'corrected_slope': corrected_slope_a},
        'model_ap_calibration': {'apparent_auc': apparent_ap['auc'], 'optimism_auc': optimism_auc_ap,
                                  'corrected_auc': corrected_auc_ap, 'apparent_brier': apparent_brier_ap_full,
                                  'corrected_brier': corrected_brier_ap,
                                  'apparent_intercept': apparent_calib_ap_full[0], 'corrected_intercept': corrected_int_ap,
                                  'apparent_slope': apparent_calib_ap_full[1], 'corrected_slope': corrected_slope_ap},
        'delta_auc_bootstrap_primary': {'median': delta_med_apparent, 'ci_lo': delta_lo_apparent,
                                         'ci_hi': delta_hi_apparent},
        'delta_auc_bootstrap_diagnostic_test_cohort': {'median': delta_med_test, 'ci_lo': delta_lo_test,
                                                        'ci_hi': delta_hi_test},
    }


# ============================================================
# 3.6 REPEATED K-FOLD (POOLED OOF, "EMPIRICAL RANGE")
# ============================================================
def repeated_kfold_pooled_oof(cohort_b, n_repeats=N_KFOLD_REPEATS, k=K_FOLDS, seed=RANDOM_SEED):
    print_header(f"3.6 REPEATED {k}-FOLD x {n_repeats} REPEAT -- pooled OOF, empirical range")
    y = cohort_b['csPCa_binary'].values
    age = cohort_b['age_per_10'].values
    psa = cohort_b['log_PSA'].values
    vol_manual = cohort_b['log_manual_volume'].values
    vol_ai = cohort_b['log_AI_volume'].values
    centre = cohort_b['centre_RUMC'].values
    n = len(y)

    auc_a_repeats, auc_ap_repeats, delta_repeats = [], [], []
    n_fold_failures = 0

    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed + rep)
        oof_pred_a = np.full(n, np.nan)
        oof_pred_ap = np.full(n, np.nan)

        for train_idx, test_idx in skf.split(age, y):
            y_tr = y[train_idx]
            if len(np.unique(y_tr)) < 2:
                n_fold_failures += 1
                continue
            try:
                X_a_tr = sm.add_constant(np.column_stack(
                    [age[train_idx], psa[train_idx], vol_manual[train_idx], centre[train_idx]]))
                model_a = fit_logit(y_tr, X_a_tr)
                X_a_te = sm.add_constant(np.column_stack(
                    [age[test_idx], psa[test_idx], vol_manual[test_idx], centre[test_idx]]),
                    has_constant='add')
                oof_pred_a[test_idx] = model_a.predict(X_a_te)

                X_ap_tr = sm.add_constant(np.column_stack(
                    [age[train_idx], psa[train_idx], vol_ai[train_idx], centre[train_idx]]))
                model_ap = fit_logit(y_tr, X_ap_tr)
                X_ap_te = sm.add_constant(np.column_stack(
                    [age[test_idx], psa[test_idx], vol_ai[test_idx], centre[test_idx]]),
                    has_constant='add')
                oof_pred_ap[test_idx] = model_ap.predict(X_ap_te)
            except Exception:
                n_fold_failures += 1
                continue

        valid = ~np.isnan(oof_pred_a) & ~np.isnan(oof_pred_ap)
        if valid.sum() < n * 0.5 or len(np.unique(y[valid])) < 2:
            print(f"  Repeat {rep}: yetersiz gecerli OOF tahmin, atlaniyor.")
            continue
        auc_a = roc_auc_score(y[valid], oof_pred_a[valid])
        auc_ap = roc_auc_score(y[valid], oof_pred_ap[valid])
        auc_a_repeats.append(auc_a)
        auc_ap_repeats.append(auc_ap)
        delta_repeats.append(auc_ap - auc_a)

    print(f"Tamamlanan repeat: {len(auc_a_repeats)}/{n_repeats} (fold-level basarisiz: {n_fold_failures})")

    def summarize(arr, name):
        arr = np.array(arr)
        med, lo, hi = np.median(arr), np.percentile(arr, 2.5), np.percentile(arr, 97.5)
        print(f"  {name}: median={med:.4f}, mean={arr.mean():.4f}+-{arr.std():.4f}, "
              f"empirical 2.5-97.5th percentile range=[{lo:.4f}, {hi:.4f}]  (BU BIR 95% CI DEGIL)")
        return {'median': med, 'mean': arr.mean(), 'sd': arr.std(), 'range_lo': lo, 'range_hi': hi}

    result_a = summarize(auc_a_repeats, "Model A (pooled-OOF AUC per repeat)")
    result_ap = summarize(auc_ap_repeats, "Model A' (pooled-OOF AUC per repeat)")
    result_delta = summarize(delta_repeats, "Delta AUC (A'-A) per repeat")

    return {'model_a': result_a, 'model_ap': result_ap, 'delta': result_delta,
            'n_repeats_completed': len(auc_a_repeats), 'n_fold_failures': n_fold_failures}


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print_header("PHASE 3: PRIMER ANALIZLER")
    df = load_raw()
    cohort_a = build_cohort_a(df)
    cohort_b = build_cohort_b(df)

    # ---- 3.1 ----
    print_header("3.1 MODEL A / MODEL A' YENIDEN FIT (Cohort B)")
    y_b = cohort_b['csPCa_binary'].values
    age = cohort_b['age_per_10'].values
    psa = cohort_b['log_PSA'].values
    vol_manual = cohort_b['log_manual_volume'].values
    vol_ai = cohort_b['log_AI_volume'].values
    centre = cohort_b['centre_RUMC'].values

    X_a = sm.add_constant(np.column_stack([age, psa, vol_manual, centre]))
    X_ap = sm.add_constant(np.column_stack([age, psa, vol_ai, centre]))
    var_names = ['Intercept', 'age_per_10', 'log_PSA', 'log_volume', 'centre_RUMC']

    apparent_a = fit_and_report_model(y_b, X_a, var_names, "Model A (manual volume)")
    apparent_ap = fit_and_report_model(y_b, X_ap, var_names, "Model A' (AI volume)")

    auc1, auc2, p_delong_models = delong_paired_test(y_b, apparent_a['pred'], apparent_ap['pred'])
    print(f"\nModel A vs A' paired DeLong test (apparent, ayni kohort): p={p_delong_models:.4g}")

    # ---- 3.2 ----
    print_header("3.2 LRT — PSAD ESIT-AGIRLIK KISITI TESTI")
    lrt_a = run_lrt(y_b, age, vol_manual, psa, centre, "Model A (manuel hacim)")
    lrt_ap = run_lrt(y_b, age, vol_ai, psa, centre, "Model A' (AI hacim)")

    # ---- 3.3 ----
    single_marker_result = run_primary_single_marker(cohort_a)

    # ---- 3.4 + 3.5 (birlesik) ----
    combined = combined_bootstrap_ratio_and_calibration(cohort_b, apparent_a, apparent_ap)

    # ---- 3.6 ----
    kfold_result = repeated_kfold_pooled_oof(cohort_b)

    # ---- CIKTI DOSYALARI ----
    summary_rows = [
        {'analysis': '3.1 Model A apparent AUC', 'value': apparent_a['auc'],
         'ci_lo': apparent_a['auc_lo'], 'ci_hi': apparent_a['auc_hi']},
        {'analysis': '3.1 Model A\' apparent AUC', 'value': apparent_ap['auc'],
         'ci_lo': apparent_ap['auc_lo'], 'ci_hi': apparent_ap['auc_hi']},
        {'analysis': '3.2 LRT Model A p', 'value': lrt_a['p_value'], 'ci_lo': None, 'ci_hi': None},
        {'analysis': '3.2 LRT Model A\' p', 'value': lrt_ap['p_value'], 'ci_lo': None, 'ci_hi': None},
        {'analysis': '3.3 calculated manual PSAD AUC', 'value': single_marker_result['auc_calculated_manual_psad'],
         'ci_lo': single_marker_result['ci_calc'][0], 'ci_hi': single_marker_result['ci_calc'][1]},
        {'analysis': '3.3 AI-PSAD AUC', 'value': single_marker_result['auc_ai_psad'],
         'ci_lo': single_marker_result['ci_ai'][0], 'ci_hi': single_marker_result['ci_ai'][1]},
        {'analysis': '3.4 coef ratio Model A (bootstrap median)', 'value': combined['ratio_a']['median'],
         'ci_lo': combined['ratio_a']['ci_lo'], 'ci_hi': combined['ratio_a']['ci_hi']},
        {'analysis': '3.4 coef ratio Model A\' (bootstrap median)', 'value': combined['ratio_ap']['median'],
         'ci_lo': combined['ratio_ap']['ci_lo'], 'ci_hi': combined['ratio_ap']['ci_hi']},
        {'analysis': '3.5 Model A optimism-corrected AUC', 'value': combined['model_a_calibration']['corrected_auc'],
         'ci_lo': None, 'ci_hi': None},
        {'analysis': '3.5 Model A\' optimism-corrected AUC', 'value': combined['model_ap_calibration']['corrected_auc'],
         'ci_lo': None, 'ci_hi': None},
        {'analysis': '3.6 repeated-kfold Model A AUC (empirical range)', 'value': kfold_result['model_a']['median'],
         'ci_lo': kfold_result['model_a']['range_lo'], 'ci_hi': kfold_result['model_a']['range_hi']},
        {'analysis': '3.6 repeated-kfold Model A\' AUC (empirical range)', 'value': kfold_result['model_ap']['median'],
         'ci_lo': kfold_result['model_ap']['range_lo'], 'ci_hi': kfold_result['model_ap']['range_hi']},
    ]
    pd.DataFrame(summary_rows).to_csv(os.path.join(OUTPUT_DIR, "phase3_primary_analyses_summary.csv"), index=False)

    print_header("PHASE 3 TAMAMLANDI")
    print(f"Ozet CSV kaydedildi: {os.path.join(OUTPUT_DIR, 'phase3_primary_analyses_summary.csv')}")
    print("Not: Phase 4'e (center-based validation) gecmeden bu sonuclar degerlendirilecek.")


if __name__ == "__main__":
    main()
