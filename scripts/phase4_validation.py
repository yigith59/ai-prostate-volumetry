"""
Revizyon Phase 4 - Validasyon
=================================================================================
Phase 3 kilitlendi (log() duzeltmesi sonrasi). Bu faz uc validasyon analizi yapiyor:

4.1  Center-based validation (Cohort B, N=1096): RUMC->ZGT ve ZGT->RUMC,
     her iki yonde de egit/test ayrimi, hakemin acikca istedigi transportability
     kontrolu.
4.2  ProstateX sensitivity (Cohort B uzerinde): ProstateX-kaynakli vakalar
     cikarilinca "kalan" (disarida degil) kohort, merkez kompozisyonu etkisi.
4.3  3-merkezli sensitivity (N=1418, PCNN dahil): supplementary, primary
     model degil.

Not: Log(x) kullanimi (log1p DEGIL) Phase 2/3 ile ayni, bu script de
kendi icinde bu kontrolu tekrarliyor (self-contained convention).
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

DATASET_ROOT = "/content/drive/MyDrive/Uroloji_Projesi/PICAI_Dataset"
MERGED_CSV_PATH = os.path.join(DATASET_ROOT, "QA_outputs", "faz2_merged_clinical_ai_volume.csv")
PROSTATEX_MAPPING_PATH = os.path.join(DATASET_ROOT, "labels", "additional_resources", "ProstateX-mapping.json")
OUTPUT_DIR = os.path.join(DATASET_ROOT, "QA_outputs", "revision_outputs")

RANDOM_SEED = 42
N_BOOTSTRAP_VALIDATION = 500
EXPECTED_N_COHORT_B = 1096


def print_header(title):
    print("\n" + "=" * 90)
    print(f"||  {title}")
    print("=" * 90)


def print_subheader(title):
    print(f"\n{'-' * 80}\n  {title}\n{'-' * 80}")


# ============================================================
# KOHORT B (Phase 1/2/3 ile birebir ayni kriter, gercek log())
# ============================================================
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

    n_nonpos = (cohort['psa'] <= 0).sum() + (cohort['prostate_volume'] <= 0).sum() + \
               (cohort['ai_volume_Bosma22b'] <= 0).sum()
    if n_nonpos > 0:
        raise ValueError("PSA/hacimde sifir/negatif deger var, gercek log() kullanilamaz.")

    cohort['log_PSA'] = np.log(cohort['psa'])
    cohort['log_manual_volume'] = np.log(cohort['prostate_volume'])
    cohort['log_AI_volume'] = np.log(cohort['ai_volume_Bosma22b'])
    cohort['centre_RUMC'] = (cohort['center'] == 'RUMC').astype(int)

    n = len(cohort)
    if n != EXPECTED_N_COHORT_B:
        print(f"UYARI: Cohort B beklenen N={EXPECTED_N_COHORT_B} degil, N={n}.")
    return cohort


def build_cohort_b_3centre(df):
    """Cohort B kriterleri, PCNN filtresi OLMADAN (Phase 1B'nin urettigi kohortla ayni)."""
    mask = (
        df['patient_age'].notna() &
        df['psa'].notna() &
        df['prostate_volume'].notna() &
        df['ai_volume_Bosma22b'].notna() &
        df['center'].notna() &
        df['case_csPCa'].notna()
    )
    cohort = df[mask].copy()
    cohort['csPCa_binary'] = (cohort['case_csPCa'] == 'YES').astype(int)
    cohort['age_per_10'] = cohort['patient_age'] / 10.0
    cohort['log_PSA'] = np.log(cohort['psa'])
    cohort['log_manual_volume'] = np.log(cohort['prostate_volume'])
    cohort['log_AI_volume'] = np.log(cohort['ai_volume_Bosma22b'])
    # ZGT referans; RUMC ve PCNN dummy
    cohort['centre_RUMC'] = (cohort['center'] == 'RUMC').astype(int)
    cohort['centre_PCNN'] = (cohort['center'] == 'PCNN').astype(int)
    return cohort


# ============================================================
# YARDIMCILAR
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
    delongcov = np.atleast_2d(sx / m + sy / n)
    return aucs, delongcov


def delong_auc_ci(y, pred, ci=0.95):
    y = np.asarray(y)
    order = np.argsort(-y)
    y_sorted = y[order]
    m = int(np.sum(y_sorted))
    if m == 0 or m == len(y):
        return np.nan, np.nan, np.nan  # tek sinif -- AUC/CI tanimsiz
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
    z = abs(aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(z))
    return aucs[0], aucs[1], p


def brier_score(y, pred):
    return float(np.mean((np.asarray(y) - np.asarray(pred)) ** 2))


def calibration_slope_intercept(y, pred, eps=1e-6):
    """N kucuk veya tek sinifsa kalibrasyon modeli yakinsamayabilir -- bu durumda
    NaN donduruluyor, sessizce yanlis sayi uretilmiyor."""
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    pred_clip = np.clip(pred, eps, 1 - eps)
    logit_pred = np.log(pred_clip / (1 - pred_clip))
    X = sm.add_constant(logit_pred)
    try:
        m = sm.Logit(y, X).fit(disp=0)
        if not m.mle_retvals.get('converged', True):
            return np.nan, np.nan
        p = np.asarray(m.params)
        return p[0], p[1]
    except Exception:
        return np.nan, np.nan


def fit_model_a_and_ap(y, age, psa, vol_manual, vol_ai, centre_cols):
    """centre_cols: 2D array (N x k), k=1 (RUMC binary) veya k=2 (RUMC+PCNN dummy)."""
    X_a = sm.add_constant(np.column_stack([age, psa, vol_manual] + [centre_cols[:, i] for i in range(centre_cols.shape[1])]))
    X_ap = sm.add_constant(np.column_stack([age, psa, vol_ai] + [centre_cols[:, i] for i in range(centre_cols.shape[1])]))
    model_a = fit_logit(y, X_a)
    model_ap = fit_logit(y, X_ap)
    return model_a, model_ap, X_a, X_ap


# ============================================================
# 4.1 CENTER-BASED VALIDATION
# ============================================================
def bootstrap_paired_delta_auc_fixed_pred(y, pred1, pred2, n_boot=N_BOOTSTRAP_VALIDATION, seed=RANDOM_SEED):
    """Sabit (zaten hesaplanmis) test-set tahminleri uzerinde paired resampling --
    center-validation test setleri kucuk oldugu icin model refit yerine sabit
    tahminlerle resampling yapmak daha stabil (refit her resample'da ayrisma riski tasir)."""
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    pred1, pred2 = np.asarray(pred1), np.asarray(pred2)
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
    if len(deltas) == 0:
        return {'n_boot': n_boot, 'n_failed': n_failed, 'median': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan}
    deltas = np.array(deltas)
    return {'n_boot': n_boot, 'n_failed': n_failed, 'median': np.median(deltas),
            'ci_lo': np.percentile(deltas, 2.5), 'ci_hi': np.percentile(deltas, 97.5)}


def run_center_validation_direction(cohort_b, train_centre, test_centre, label):
    print_subheader(f"{label}: train={train_centre}, test={test_centre}")

    train = cohort_b[cohort_b['center'] == train_centre]
    test = cohort_b[cohort_b['center'] == test_centre]

    n_train, events_train = len(train), int(train['csPCa_binary'].sum())
    n_test, events_test = len(test), int(test['csPCa_binary'].sum())
    print(f"n_train={n_train}, events_train={events_train} ({100*events_train/n_train:.1f}%)")
    print(f"n_test={n_test}, events_test={events_test} ({100*events_test/n_test:.1f}%)")

    y_train = train['csPCa_binary'].values
    # centre train setinde SABIT (tek merkez) oldugu icin model formuluna centre KATILMIYOR --
    # aksi halde perfect-separation/singularite olurdu (centre bir kolonun tamamini tek deger yapardi).
    X_a_train = sm.add_constant(np.column_stack(
        [train['age_per_10'].values, train['log_PSA'].values, train['log_manual_volume'].values]))
    X_ap_train = sm.add_constant(np.column_stack(
        [train['age_per_10'].values, train['log_PSA'].values, train['log_AI_volume'].values]))

    try:
        model_a = fit_logit(y_train, X_a_train)
        model_ap = fit_logit(y_train, X_ap_train)
    except RuntimeError as e:
        print(f"  UYARI: model egitim setinde yakinsamadi ({e}) -- bu yon atlaniyor.")
        return None

    y_test = test['csPCa_binary'].values
    X_a_test = sm.add_constant(np.column_stack(
        [test['age_per_10'].values, test['log_PSA'].values, test['log_manual_volume'].values]), has_constant='add')
    X_ap_test = sm.add_constant(np.column_stack(
        [test['age_per_10'].values, test['log_PSA'].values, test['log_AI_volume'].values]), has_constant='add')

    pred_a_test = model_a.predict(X_a_test)
    pred_ap_test = model_ap.predict(X_ap_test)

    auc_a, auc_a_lo, auc_a_hi = delong_auc_ci(y_test, pred_a_test)
    auc_ap, auc_ap_lo, auc_ap_hi = delong_auc_ci(y_test, pred_ap_test)
    brier_a = brier_score(y_test, pred_a_test)
    brier_ap = brier_score(y_test, pred_ap_test)
    calib_a = calibration_slope_intercept(y_test, pred_a_test)
    calib_ap = calibration_slope_intercept(y_test, pred_ap_test)

    print(f"Model A  test AUC={auc_a:.4f} (DeLong CI {auc_a_lo:.4f}-{auc_a_hi:.4f}), Brier={brier_a:.4f}, "
          f"calib intercept={calib_a[0]:.4f}, slope={calib_a[1]:.4f}")
    print(f"Model A' test AUC={auc_ap:.4f} (DeLong CI {auc_ap_lo:.4f}-{auc_ap_hi:.4f}), Brier={brier_ap:.4f}, "
          f"calib intercept={calib_ap[0]:.4f}, slope={calib_ap[1]:.4f}")

    if np.isnan(calib_a[1]) or np.isnan(calib_ap[1]):
        print("  UYARI: kalibrasyon modeli bu test setinde yakinsamadi/tanimsiz -- kucuk N/az event "
              "nedeniyle olabilir, sonuc NaN olarak birakildi, gizlenmedi.")

    boot = bootstrap_paired_delta_auc_fixed_pred(y_test, pred_a_test, pred_ap_test)
    print(f"Bootstrap paired deltaAUC (B={boot['n_boot']}, {boot['n_failed']} basarisiz): "
          f"median={boot['median']:.4f}, 95% CI [{boot['ci_lo']:.4f}, {boot['ci_hi']:.4f}]")

    # Ek (2026-09-10): train setinde fit edilen modellerin coefficient ratio'su --
    # descriptive, bootstrap CI YOK (sadece yon tutarliligi kontrolu icin istendi).
    # Kolon sirasi: const(0), age(1), psa(2), vol(3) -- centre train setinde sabit
    # oldugu icin modelde yok.
    params_a = np.asarray(model_a.params)
    params_ap = np.asarray(model_ap.params)
    ratio_a_train = abs(params_a[3] / params_a[2])
    ratio_ap_train = abs(params_ap[3] / params_ap[2])
    print(f"[Descriptive, CI yok] {train_centre}-trained coefficient ratio: "
          f"Model A={ratio_a_train:.4f}, Model A'={ratio_ap_train:.4f}")

    return {
        'direction': label, 'n_train': n_train, 'events_train': events_train,
        'n_test': n_test, 'events_test': events_test,
        'auc_a': auc_a, 'auc_a_ci': (auc_a_lo, auc_a_hi), 'brier_a': brier_a, 'calib_a': calib_a,
        'auc_ap': auc_ap, 'auc_ap_ci': (auc_ap_lo, auc_ap_hi), 'brier_ap': brier_ap, 'calib_ap': calib_ap,
        'delta_auc_bootstrap': boot,
        'ratio_a_train': ratio_a_train, 'ratio_ap_train': ratio_ap_train,
    }


# ============================================================
# 4.2 PROSTATEX SENSITIVITY
# ============================================================
def load_prostatex_case_ids():
    """ProstateX-mapping.json: {"ProstateX-XXXX_tarih": "patient_id_study_id"} --
    degerler dogrudan case_id formatiyla eslesiyor (Faz 2c'deki ayni mantik)."""
    import json
    if not os.path.exists(PROSTATEX_MAPPING_PATH):
        print(f"UYARI: {PROSTATEX_MAPPING_PATH} bulunamadi, ProstateX sensitivity atlanacak.")
        return set()
    with open(PROSTATEX_MAPPING_PATH) as f:
        mapping = json.load(f)
    case_ids = set(mapping.values())
    print(f"ProstateX-mapping.json: {len(mapping)} eslesme, {len(case_ids)} benzersiz case_id.")
    return case_ids


def run_prostatex_sensitivity(cohort_b):
    print_header("4.2 PROSTATEX SENSITIVITY (Cohort B, N=%d)" % len(cohort_b))
    prostatex_ids = load_prostatex_case_ids()
    if len(prostatex_ids) == 0:
        return None

    is_prostatex = cohort_b['case_id'].isin(prostatex_ids)
    n_prostatex_in_b = int(is_prostatex.sum())
    print(f"Cohort B'de ProstateX-kaynakli vaka: {n_prostatex_in_b}/{len(cohort_b)}")

    print("ProstateX vakalarinin merkez dagilimi (Cohort B icinde):")
    for c in sorted(cohort_b.loc[is_prostatex, 'center'].unique()):
        n_c = int((cohort_b.loc[is_prostatex, 'center'] == c).sum())
        print(f"  {c}: {n_c}")

    remaining = cohort_b[~is_prostatex].copy()
    n_remaining = len(remaining)
    events_remaining = int(remaining['csPCa_binary'].sum())
    print(f"\nKALAN kohort (dislanan degil): N={n_remaining}, events={events_remaining} "
          f"({100*events_remaining/n_remaining:.1f}%)")
    print("Center dagilimi (kalan kohort):")
    for c in sorted(remaining['center'].unique()):
        n_c = int((remaining['center'] == c).sum())
        print(f"  {c}: {n_c} ({100*n_c/n_remaining:.1f}%)")

    y = remaining['csPCa_binary'].values
    age = remaining['age_per_10'].values
    psa = remaining['log_PSA'].values
    vol_m = remaining['log_manual_volume'].values
    vol_ai = remaining['log_AI_volume'].values
    centre = remaining['centre_RUMC'].values.reshape(-1, 1)

    model_a, model_ap, X_a, X_ap = fit_model_a_and_ap(y, age, psa, vol_m, vol_ai, centre)
    pred_a, pred_ap = model_a.predict(X_a), model_ap.predict(X_ap)

    auc_a, lo_a, hi_a = delong_auc_ci(y, pred_a)
    auc_ap, lo_ap, hi_ap = delong_auc_ci(y, pred_ap)
    _, _, p_delong = delong_paired_test(y, pred_a, pred_ap)
    print(f"\nModel A  AUC={auc_a:.4f} (DeLong CI {lo_a:.4f}-{hi_a:.4f})")
    print(f"Model A' AUC={auc_ap:.4f} (DeLong CI {lo_ap:.4f}-{hi_ap:.4f})")
    print(f"Paired DeLong p={p_delong:.4g}")

    boot = bootstrap_paired_delta_auc_fixed_pred(y, pred_a, pred_ap)
    print(f"Bootstrap paired deltaAUC (B={boot['n_boot']}, {boot['n_failed']} basarisiz): "
          f"median={boot['median']:.4f}, 95% CI [{boot['ci_lo']:.4f}, {boot['ci_hi']:.4f}]")

    return {
        'n_prostatex_in_cohort_b': n_prostatex_in_b, 'n_remaining': n_remaining,
        'events_remaining': events_remaining, 'auc_a': auc_a, 'auc_ap': auc_ap,
        'delong_p': p_delong, 'delta_auc_bootstrap': boot,
    }


# ============================================================
# 4.3 3-MERKEZLI SENSITIVITY
# ============================================================
def run_3centre_sensitivity(df, primary_delta_auc):
    print_header("4.3 3-MERKEZLI SENSITIVITY (N~1418, PCNN dahil, SUPPLEMENTARY)")
    cohort = build_cohort_b_3centre(df)
    n = len(cohort)
    events = int(cohort['csPCa_binary'].sum())
    print(f"N={n}, events={events} ({100*events/n:.1f}%)")
    for c in sorted(cohort['center'].unique()):
        n_c = int((cohort['center'] == c).sum())
        e_c = int(cohort.loc[cohort['center'] == c, 'csPCa_binary'].sum())
        print(f"  {c}: N={n_c}, events={e_c} ({100*e_c/n_c:.1f}%)")

    y = cohort['csPCa_binary'].values
    age = cohort['age_per_10'].values
    psa = cohort['log_PSA'].values
    vol_m = cohort['log_manual_volume'].values
    vol_ai = cohort['log_AI_volume'].values
    centre_cols = cohort[['centre_RUMC', 'centre_PCNN']].values  # ZGT referans

    var_names = ['Intercept', 'age_per_10', 'log_PSA', 'log_volume', 'centre_RUMC_vs_ZGT', 'centre_PCNN_vs_ZGT']

    model_a, model_ap, X_a, X_ap = fit_model_a_and_ap(y, age, psa, vol_m, vol_ai, centre_cols)
    pred_a, pred_ap = model_a.predict(X_a), model_ap.predict(X_ap)

    for label, model, pred in [("Model A", model_a, pred_a), ("Model A'", model_ap, pred_ap)]:
        auc, lo, hi = delong_auc_ci(y, pred)
        print(f"\n{label}: AIC={model.aic:.3f}, AUC={auc:.4f} (DeLong CI {lo:.4f}-{hi:.4f})")
        params, ses, pvals = np.asarray(model.params), np.asarray(model.bse), np.asarray(model.pvalues)
        for i, name in enumerate(var_names):
            if name == 'Intercept':
                continue
            or_val = np.exp(params[i])
            print(f"  {name}: coef={params[i]:.4f}, OR={or_val:.3f}, p={pvals[i]:.4g}")

    _, _, p_delong = delong_paired_test(y, pred_a, pred_ap)
    auc_a_val, _, _ = delong_auc_ci(y, pred_a)
    auc_ap_val, _, _ = delong_auc_ci(y, pred_ap)
    delta_auc_3c = auc_ap_val - auc_a_val
    print(f"\nDeLong p (Model A vs A')={p_delong:.4g}, deltaAUC={delta_auc_3c:.4f}")

    # Ek (2026-09-10): 3-merkezli sensitivity icin paired bootstrap deltaAUC CI --
    # standart resample-the-statistic yontemi (Phase 3'teki PRIMARY yontemle ayni mantik):
    # her resample'da Model A/A' YENIDEN FIT edilir, kendi resample'inda degerlendirilir.
    rng = np.random.RandomState(RANDOM_SEED)
    n_obs = len(y)
    deltas = []
    n_failed_boot = 0
    for _ in range(N_BOOTSTRAP_VALIDATION):
        idx = rng.randint(0, n_obs, n_obs)
        y_b = y[idx]
        if len(np.unique(y_b)) < 2:
            n_failed_boot += 1
            continue
        try:
            age_b, psa_b, vm_b, va_b = age[idx], psa[idx], vol_m[idx], vol_ai[idx]
            centre_b = centre_cols[idx]
            model_a_b, model_ap_b, X_a_b, X_ap_b = fit_model_a_and_ap(y_b, age_b, psa_b, vm_b, va_b, centre_b)
            pred_a_b = model_a_b.predict(X_a_b)
            pred_ap_b = model_ap_b.predict(X_ap_b)
            deltas.append(roc_auc_score(y_b, pred_ap_b) - roc_auc_score(y_b, pred_a_b))
        except Exception:
            n_failed_boot += 1
    if len(deltas) > 0:
        deltas = np.array(deltas)
        print(f"Bootstrap paired deltaAUC (B={N_BOOTSTRAP_VALIDATION}, {n_failed_boot} basarisiz): "
              f"median={np.median(deltas):.4f}, 95% CI [{np.percentile(deltas, 2.5):.4f}, "
              f"{np.percentile(deltas, 97.5):.4f}]")
    else:
        print("UYARI: 3-merkezli bootstrap hicbir resample'da basarili olmadi.")

    if primary_delta_auc is not None:
        same_direction = (delta_auc_3c > 0) == (primary_delta_auc > 0)
        print(f"\nCohort B primary (N=1096) deltaAUC={primary_delta_auc:.4f} ile karsilastirma: "
              f"{'YON TUTARLI (directionally consistent)' if same_direction else 'YON TUTARSIZ -- FARKLI!'}")

    return {'n': n, 'events': events, 'delta_auc': delta_auc_3c, 'delong_p': p_delong}


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print_header("PHASE 4: VALIDASYON")
    df = pd.read_csv(MERGED_CSV_PATH)
    cohort_b = build_cohort_b(df)

    # ---- 4.1 ----
    print_header("4.1 CENTER-BASED VALIDATION (Cohort B, N=%d)" % len(cohort_b))
    result_rumc_zgt = run_center_validation_direction(cohort_b, 'RUMC', 'ZGT', "Senaryo 1")
    result_zgt_rumc = run_center_validation_direction(cohort_b, 'ZGT', 'RUMC', "Senaryo 2")

    # ---- 4.2 ----
    prostatex_result = run_prostatex_sensitivity(cohort_b)

    # ---- 4.3 (Cohort B primary deltaAUC referans olarak Phase 3'ten aliniyor) ----
    PRIMARY_DELTA_AUC_COHORT_B = 0.0145  # Phase 3.1'den (Model A' apparent - Model A apparent)
    result_3centre = run_3centre_sensitivity(df, PRIMARY_DELTA_AUC_COHORT_B)

    # ---- OZET CSV ----
    rows = []
    for r, name in [(result_rumc_zgt, 'RUMC->ZGT'), (result_zgt_rumc, 'ZGT->RUMC')]:
        if r is None:
            rows.append({'analysis': name, 'status': 'FAILED_TO_CONVERGE'})
            continue
        rows.append({
            'analysis': name, 'n_train': r['n_train'], 'events_train': r['events_train'],
            'n_test': r['n_test'], 'events_test': r['events_test'],
            'auc_a': r['auc_a'], 'auc_ap': r['auc_ap'],
            'delta_auc_bootstrap_median': r['delta_auc_bootstrap']['median'],
            'delta_auc_ci_lo': r['delta_auc_bootstrap']['ci_lo'],
            'delta_auc_ci_hi': r['delta_auc_bootstrap']['ci_hi'],
            'calib_slope_a': r['calib_a'][1], 'calib_slope_ap': r['calib_ap'][1],
        })
    if prostatex_result is not None:
        rows.append({
            'analysis': 'ProstateX sensitivity (kalan kohort)',
            'n_test': prostatex_result['n_remaining'], 'events_test': prostatex_result['events_remaining'],
            'auc_a': prostatex_result['auc_a'], 'auc_ap': prostatex_result['auc_ap'],
            'delta_auc_bootstrap_median': prostatex_result['delta_auc_bootstrap']['median'],
            'delta_auc_ci_lo': prostatex_result['delta_auc_bootstrap']['ci_lo'],
            'delta_auc_ci_hi': prostatex_result['delta_auc_bootstrap']['ci_hi'],
        })
    rows.append({
        'analysis': '3-centre sensitivity (supplementary)',
        'n_test': result_3centre['n'], 'events_test': result_3centre['events'],
        'delta_auc_bootstrap_median': result_3centre['delta_auc'],
    })

    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "phase4_validation_summary.csv"), index=False)

    print_header("PHASE 4 TAMAMLANDI")
    print(f"Ozet CSV kaydedildi: {os.path.join(OUTPUT_DIR, 'phase4_validation_summary.csv')}")


if __name__ == "__main__":
    main()
