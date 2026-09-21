"""
Revizyon Phase 5 - Klinik Performans
=================================================================================
5.1  PSAD threshold performansi (Cohort A, N=1418): calculated manual PSAD
     ve AI-PSAD icin 0.10/0.15/0.20.
5.2  Reclassification tablosu (threshold=0.15): manual vs AI-PSAD.
5.3  Model A/A' predicted-probability threshold performansi (Cohort B,
     N=1096), REPEATED-CV ORTALAMALI OOF tahminlerle (apparent degil).
5.4  DCA (Cohort B), OOF tahminlerle, bootstrap CI'li, klinik dile cevrilmis
     net benefit farki.

KRITIK TASARIM KARARI: Phase 3.6'da 5-fold x 20-repeat calisti ama OOF
tahminlerin KENDISI kaydedilmedi (sadece repeat-basi AUC). Bu script o
k-fold prosedurunu TEKRAR calistiriyor, ama bu sefer her hastanin 20
repeat'teki OOF tahminini ORTALAYARAK tek, stabil bir "repeated-CV OOF"
tahmin seti uretiyor -- bu, threshold/DCA analizleri icin tek bir rastgele
5-fold split'ine guvenmekten daha guvenilir.
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

DATASET_ROOT = "/content/drive/MyDrive/Uroloji_Projesi/PICAI_Dataset"
MERGED_CSV_PATH = os.path.join(DATASET_ROOT, "QA_outputs", "faz2_merged_clinical_ai_volume.csv")
OUTPUT_DIR = os.path.join(DATASET_ROOT, "QA_outputs", "revision_outputs")

RANDOM_SEED = 42
N_KFOLD_REPEATS = 20
K_FOLDS = 5
N_BOOTSTRAP_DCA = 500
PSAD_THRESHOLDS = [0.10, 0.15, 0.20]
MODEL_THRESHOLDS = [0.10, 0.15, 0.20, 0.30]
DCA_THRESHOLD_RANGE = np.arange(0.05, 0.51, 0.01)
DCA_REPORT_THRESHOLDS = [0.10, 0.15, 0.20, 0.30]


def print_header(title):
    print("\n" + "=" * 90)
    print(f"||  {title}")
    print("=" * 90)


def print_subheader(title):
    print(f"\n{'-' * 80}\n  {title}\n{'-' * 80}")


# ============================================================
# KOHORTLAR (Phase 1/3/4 ile birebir ayni)
# ============================================================
def build_cohort_a(df):
    mask = (
        df['psa'].notna() & df['prostate_volume'].notna() &
        df['ai_volume_Bosma22b'].notna() & df['case_csPCa'].notna()
    )
    cohort = df[mask].copy()
    cohort['csPCa_binary'] = (cohort['case_csPCa'] == 'YES').astype(int)
    cohort['calculated_manual_psad'] = cohort['psa'] / cohort['prostate_volume']
    cohort['ai_psad'] = cohort['psa'] / cohort['ai_volume_Bosma22b']
    return cohort


def build_cohort_b(df):
    mask = (
        df['patient_age'].notna() & df['psa'].notna() & df['prostate_volume'].notna() &
        df['ai_volume_Bosma22b'].notna() & df['center'].notna() & df['case_csPCa'].notna() &
        (df['center'] != 'PCNN')
    )
    cohort = df[mask].copy()
    cohort['csPCa_binary'] = (cohort['case_csPCa'] == 'YES').astype(int)
    cohort['age_per_10'] = cohort['patient_age'] / 10.0
    cohort['log_PSA'] = np.log(cohort['psa'])
    cohort['log_manual_volume'] = np.log(cohort['prostate_volume'])
    cohort['log_AI_volume'] = np.log(cohort['ai_volume_Bosma22b'])
    cohort['centre_RUMC'] = (cohort['center'] == 'RUMC').astype(int)
    return cohort


def fit_logit(y, X):
    model = sm.Logit(y, X).fit(disp=0)
    if not model.mle_retvals.get('converged', True):
        raise RuntimeError("Model yakinsamadi.")
    return model


# ============================================================
# 5.1 / 5.3 THRESHOLD METRIKLERI (ORTAK FONKSIYON)
# ============================================================
def threshold_metrics(y, score, threshold):
    y = np.asarray(y)
    positive = np.asarray(score) >= threshold
    tp = int(np.sum(positive & (y == 1)))
    fp = int(np.sum(positive & (y == 0)))
    tn = int(np.sum(~positive & (y == 0)))
    fn = int(np.sum(~positive & (y == 1)))
    n = len(y)

    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    biopsies_avoided_per_100 = (tn + fn) / n * 100
    missed_per_100 = fn / n * 100

    return {'threshold': threshold, 'n': n, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'sensitivity': sens, 'specificity': spec, 'ppv': ppv, 'npv': npv,
            'biopsies_avoided_per_100': biopsies_avoided_per_100, 'missed_csPCa_per_100': missed_per_100}


def print_threshold_table(rows, title):
    print_subheader(title)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


# ============================================================
# 5.1 PSAD THRESHOLD PERFORMANSI (Cohort A)
# ============================================================
def run_psad_threshold_analysis(cohort_a):
    print_header("5.1 PSAD THRESHOLD PERFORMANSI (Cohort A, N=%d)" % len(cohort_a))
    y = cohort_a['csPCa_binary'].values
    rows = []
    for marker_name, marker_col in [('calculated_manual_psad', 'Calculated manual PSAD'),
                                     ('ai_psad', 'AI-PSAD')]:
        score = cohort_a[marker_name].values
        for t in PSAD_THRESHOLDS:
            m = threshold_metrics(y, score, t)
            m['marker'] = marker_col
            rows.append(m)
    df = print_threshold_table(rows, "Calculated manual PSAD vs AI-PSAD -- 0.10/0.15/0.20")
    return df


# ============================================================
# 5.2 RECLASSIFICATION TABLOSU
# ============================================================
def run_reclassification(cohort_a, threshold=0.15):
    print_header(f"5.2 RECLASSIFICATION TABLOSU (threshold={threshold}, Cohort A, N=%d)" % len(cohort_a))
    y = cohort_a['csPCa_binary'].values
    manual_pos = cohort_a['calculated_manual_psad'].values >= threshold
    ai_pos = cohort_a['ai_psad'].values >= threshold

    cells = []
    for manual_label, manual_mask in [('manual >= threshold', manual_pos), ('manual < threshold', ~manual_pos)]:
        for ai_label, ai_mask in [('AI >= threshold', ai_pos), ('AI < threshold', ~ai_pos)]:
            mask = manual_mask & ai_mask
            n_total = int(mask.sum())
            n_pos = int((mask & (y == 1)).sum())
            n_neg = int((mask & (y == 0)).sum())
            cells.append({'manual_PSAD': manual_label, 'AI_PSAD': ai_label,
                           'n_total': n_total, 'csPCa_positive': n_pos, 'csPCa_negative': n_neg})

    df = pd.DataFrame(cells)
    print(df.to_string(index=False))

    n_ai_downgrade = df.loc[(df['manual_PSAD'] == 'manual >= threshold') &
                             (df['AI_PSAD'] == 'AI < threshold'), 'n_total'].values
    n_ai_upgrade = df.loc[(df['manual_PSAD'] == 'manual < threshold') &
                           (df['AI_PSAD'] == 'AI >= threshold'), 'n_total'].values
    print(f"\nManuel>=t, AI<t (AI daha az biyopsi onerdi): n={n_ai_downgrade[0] if len(n_ai_downgrade) else 0}")
    print(f"Manuel<t, AI>=t (AI daha fazla biyopsi onerdi): n={n_ai_upgrade[0] if len(n_ai_upgrade) else 0}")

    return df


# ============================================================
# 5.3 REPEATED-CV ORTALAMALI OOF TAHMINLERI (Model A / A')
# ============================================================
def generate_averaged_oof_predictions(cohort_b, n_repeats=N_KFOLD_REPEATS, k=K_FOLDS, seed=RANDOM_SEED):
    print_header(f"OOF tahmin uretimi -- {k}-fold x {n_repeats} repeat, hasta-basi ORTALAMA")
    y = cohort_b['csPCa_binary'].values
    age = cohort_b['age_per_10'].values
    psa = cohort_b['log_PSA'].values
    vol_manual = cohort_b['log_manual_volume'].values
    vol_ai = cohort_b['log_AI_volume'].values
    centre = cohort_b['centre_RUMC'].values
    n = len(y)

    sum_pred_a = np.zeros(n)
    sum_pred_ap = np.zeros(n)
    count_a = np.zeros(n)
    count_ap = np.zeros(n)
    n_fold_failures = 0

    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed + rep)
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
                    [age[test_idx], psa[test_idx], vol_manual[test_idx], centre[test_idx]]), has_constant='add')
                sum_pred_a[test_idx] += model_a.predict(X_a_te)
                count_a[test_idx] += 1

                X_ap_tr = sm.add_constant(np.column_stack(
                    [age[train_idx], psa[train_idx], vol_ai[train_idx], centre[train_idx]]))
                model_ap = fit_logit(y_tr, X_ap_tr)
                X_ap_te = sm.add_constant(np.column_stack(
                    [age[test_idx], psa[test_idx], vol_ai[test_idx], centre[test_idx]]), has_constant='add')
                sum_pred_ap[test_idx] += model_ap.predict(X_ap_te)
                count_ap[test_idx] += 1
            except Exception:
                n_fold_failures += 1
                continue

    print(f"Fold-level basarisiz: {n_fold_failures}")
    if np.any(count_a == 0) or np.any(count_ap == 0):
        n_never_predicted = int(np.sum((count_a == 0) | (count_ap == 0)))
        print(f"UYARI: {n_never_predicted} hasta hicbir repeat'te OOF tahmin almadi (cok nadir olmali).")

    valid = (count_a > 0) & (count_ap > 0)
    avg_pred_a = np.full(n, np.nan)
    avg_pred_ap = np.full(n, np.nan)
    avg_pred_a[valid] = sum_pred_a[valid] / count_a[valid]
    avg_pred_ap[valid] = sum_pred_ap[valid] / count_ap[valid]

    print(f"Gecerli OOF tahmini olan hasta: {valid.sum()}/{n}")
    return avg_pred_a, avg_pred_ap, valid


def run_model_threshold_analysis(cohort_b, pred_a, pred_ap, valid):
    print_header("5.3 MODEL A/A' THRESHOLD PERFORMANSI (Cohort B, OOF tahminler)")
    y = cohort_b['csPCa_binary'].values[valid]
    pred_a_v, pred_ap_v = pred_a[valid], pred_ap[valid]

    rows = []
    for label, score in [('Model A (manual volume, OOF)', pred_a_v), ("Model A' (AI volume, OOF)", pred_ap_v)]:
        for t in MODEL_THRESHOLDS:
            m = threshold_metrics(y, score, t)
            m['model'] = label
            rows.append(m)
    df = print_threshold_table(rows, "Model A vs Model A' -- 0.10/0.15/0.20/0.30 (OOF)")
    return df


# ============================================================
# 5.4 DCA
# ============================================================
def net_benefit_curve(y, pred, thresholds):
    y = np.asarray(y)
    pred = np.asarray(pred)
    n = len(y)
    nb = np.zeros_like(thresholds)
    for i, pt in enumerate(thresholds):
        positive = pred >= pt
        tp = np.sum(positive & (y == 1))
        fp = np.sum(positive & (y == 0))
        nb[i] = tp / n - (fp / n) * (pt / (1 - pt))
    return nb


def treat_all_curve(prevalence, thresholds):
    return prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))


def bootstrap_dca_at_thresholds(y, pred_a, pred_ap, thresholds, n_boot=N_BOOTSTRAP_DCA, seed=RANDOM_SEED):
    """Sabit OOF tahminler uzerinde hasta resampling (refit YOK -- tahminler zaten
    disaridan/OOF olarak sabit, sadece orneklem belirsizligi olculuyor)."""
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    n = len(y)
    results = {t: {'nb_a': [], 'nb_ap': [], 'delta': []} for t in thresholds}
    n_failed = 0
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2:
            n_failed += 1
            continue
        nb_a_b = net_benefit_curve(y[idx], pred_a[idx], np.array(thresholds))
        nb_ap_b = net_benefit_curve(y[idx], pred_ap[idx], np.array(thresholds))
        for i, t in enumerate(thresholds):
            results[t]['nb_a'].append(nb_a_b[i])
            results[t]['nb_ap'].append(nb_ap_b[i])
            results[t]['delta'].append(nb_ap_b[i] - nb_a_b[i])
    summary = {}
    for t in thresholds:
        d = np.array(results[t]['delta'])
        summary[t] = {'median': np.median(d), 'ci_lo': np.percentile(d, 2.5), 'ci_hi': np.percentile(d, 97.5)}
    return summary, n_failed


def run_dca(cohort_b, pred_a, pred_ap, valid):
    print_header("5.4 DECISION CURVE ANALYSIS (Cohort B, OOF tahminler)")
    y = cohort_b['csPCa_binary'].values[valid]
    pred_a_v, pred_ap_v = pred_a[valid], pred_ap[valid]
    n = len(y)
    prevalence = y.mean()
    print(f"N={n}, prevalans={prevalence:.3f}")

    nb_a = net_benefit_curve(y, pred_a_v, DCA_THRESHOLD_RANGE)
    nb_ap = net_benefit_curve(y, pred_ap_v, DCA_THRESHOLD_RANGE)
    nb_all = treat_all_curve(prevalence, DCA_THRESHOLD_RANGE)

    boot_summary, n_failed = bootstrap_dca_at_thresholds(y, pred_a_v, pred_ap_v, DCA_REPORT_THRESHOLDS)

    print(f"\n{'pt':>6} | {'NB Model A':>12} | {'NB Model A_':>12} | {'Delta NB':>10} | "
          f"{'Boot 95% CI':>22} | {'Treat-all':>10}")
    rows = []
    for pt in DCA_REPORT_THRESHOLDS:
        idx = np.argmin(np.abs(DCA_THRESHOLD_RANGE - pt))
        delta_point = nb_ap[idx] - nb_a[idx]
        b = boot_summary[pt]
        print(f"{pt:>6.2f} | {nb_a[idx]:>12.4f} | {nb_ap[idx]:>12.4f} | {delta_point:>10.4f} | "
              f"[{b['ci_lo']:.4f}, {b['ci_hi']:.4f}]{'':>4} | {nb_all[idx]:>10.4f}")

        # Klinik ceviri: Vickers & Elkin formulu -- delta_NB * (1-pt)/pt * 100 =
        # "100 hastada net kac gereksiz biyopsi azaldi", SADECE ayni/daha iyi
        # sensitivity/FN sayisinda gecerli bir yorum -- asagida FN kontrolu ile
        # dogrulanip cumle ona gore kuruluyor.
        pos_a = pred_a_v >= pt
        pos_ap = pred_ap_v >= pt
        fn_a = int(np.sum(~pos_a & (y == 1)))
        fn_ap = int(np.sum(~pos_ap & (y == 1)))
        reduction_per_100 = delta_point * (1 - pt) / pt * 100

        if fn_ap <= fn_a:
            print(f"       -> Klinik ceviri: threshold {pt:.2f}'de Model A' yaklasik "
                  f"{reduction_per_100:.1f} daha az gereksiz biyopsiye karsilik geliyor (100 hastada), "
                  f"AYNI VEYA DAHA AZ kacan csPCa ile (FN: A={fn_a}, A'={fn_ap}).")
        else:
            print(f"       -> UYARI: threshold {pt:.2f}'de Model A' FN sayisi DAHA YUKSEK "
                  f"(A={fn_a}, A'={fn_ap}) -- 'missing additional csPCa olmadan' ifadesi BURADA "
                  f"KULLANILAMAZ, klinik ceviri cumlesi yazilmiyor.")

        rows.append({'threshold': pt, 'nb_model_a': nb_a[idx], 'nb_model_ap': nb_ap[idx],
                     'delta_nb': delta_point, 'boot_ci_lo': b['ci_lo'], 'boot_ci_hi': b['ci_hi'],
                     'nb_treat_all': nb_all[idx], 'fn_a': fn_a, 'fn_ap': fn_ap,
                     'reduction_biopsies_per_100': reduction_per_100})

    print(f"\nBootstrap: B={N_BOOTSTRAP_DCA}, basarisiz={n_failed}")
    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print_header("PHASE 5: KLINIK PERFORMANS")
    df = pd.read_csv(MERGED_CSV_PATH)

    cohort_a = build_cohort_a(df)
    cohort_b = build_cohort_b(df)

    psad_threshold_df = run_psad_threshold_analysis(cohort_a)
    reclass_df = run_reclassification(cohort_a, threshold=0.15)

    pred_a, pred_ap, valid = generate_averaged_oof_predictions(cohort_b)
    model_threshold_df = run_model_threshold_analysis(cohort_b, pred_a, pred_ap, valid)
    dca_df = run_dca(cohort_b, pred_a, pred_ap, valid)

    psad_threshold_df.to_csv(os.path.join(OUTPUT_DIR, "phase5_psad_threshold_metrics.csv"), index=False)
    reclass_df.to_csv(os.path.join(OUTPUT_DIR, "phase5_reclassification_table.csv"), index=False)
    model_threshold_df.to_csv(os.path.join(OUTPUT_DIR, "phase5_model_threshold_metrics.csv"), index=False)
    dca_df.to_csv(os.path.join(OUTPUT_DIR, "phase5_dca_results.csv"), index=False)

    print_header("PHASE 5 TAMAMLANDI")
    print(f"Ciktilar kaydedildi: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
