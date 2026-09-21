"""
Revizyon Phase 7 - Ana Metin ve Supplementary Tablo/Figur Uretimi
=================================================================================
KURAL: Hicbir sayi hardcode edilmiyor. Bu script Phase 3/4/5 scriptlerini
MODUL olarak import edip (RANDOM_SEED=42 sabit, deterministik) onlarin
zaten test edilmis/kilitlenmis fonksiyonlarini DOGRUDAN CAGIRIYOR -- boylece
uretilen her sayi, konsolda daha once dogrulanmis sayilarla otomatik olarak
birebir ayni cikiyor (ayni kod yolu, ayni seed). Legacy/out-of-scope analizler
(ADC heterojenite, lezyon radyomigi) icin gercek dosya bulunamazsa sayi ICAT
EDILMIYOR, "[TO INSERT]" olarak isaretleniyor.

Uretilen ciktilar:
  ANA METIN (5 oge):
    Figure 1  - participant flow diagram (TIFF, 600dpi)
    Figure 2  - ROC egrileri: Cohort A marker'lari + Cohort B modelleri (TIFF, 600dpi)
    Table  1  - baseline characteristics (Cohort A + Cohort B footnote paneli)
    Table  2  - discrimination + validation + calibration ozet tablosu
    Table  3  - Model A/A' katsayilari + LRT + coefficient ratio

  SUPPLEMENTARY (3 tablo + 2 figur):
    Table S1  - cohort derivation + centre distribution + reported-PSAD audit
    Table S2  - threshold performansi + reclassification + DCA ozeti
    Table S3  - centre validation + ProstateX + 3-centre + functional-form + exploratory
    Figure S1 - Bland-Altman hacim uyumu
    Figure S2 - calibration egrileri + decision-curve analysis

Her tablo icin HEM ham/granuler CSV HEM okunabilir Markdown uretilir; Markdown
ham CSV'den turetilir (elle sayi yazilmiyor). Figurler TIFF, 600 dpi, LZW
sikistirma ile kaydedilir.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve, roc_auc_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PATH AYARLARI (proje konvansiyonu: ON_COLAB oto-tespit)
# ============================================================
ON_COLAB = os.path.exists("/content/drive")
DATASET_ROOT = "/content/drive/MyDrive/Uroloji_Projesi/PICAI_Dataset" if ON_COLAB else "./PICAI_Dataset"
MERGED_CSV_PATH = os.path.join(DATASET_ROOT, "QA_outputs", "faz2_merged_clinical_ai_volume.csv")
OUTPUT_DIR = os.path.join(DATASET_ROOT, "QA_outputs", "revision_outputs")
TABLES_DIR = os.path.join(OUTPUT_DIR, "final_tables")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "final_figures")
PROSTATEX_MAPPING_PATH = os.path.join(DATASET_ROOT, "labels", "additional_resources", "ProstateX-mapping.json")
LEGACY_TRACK_A_CSV = os.path.join(OUTPUT_DIR, "faz3_track_a_results.csv")
LEGACY_TRACK_B_CSV = os.path.join(OUTPUT_DIR, "faz3_track_b_results.csv")

# Phase 1-5 scriptlerinin bulundugu klasor (Colab'da /content'e yuklendigi
# varsayiliyor -- Colab disinda script'in kendi dizini kullanilir).
PHASE_SCRIPTS_DIR = "/content" if ON_COLAB else os.path.dirname(os.path.abspath(__file__))
if PHASE_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, PHASE_SCRIPTS_DIR)

import phase3_primary_analyses as ph3
import phase4_validation as ph4
import phase5_clinical_performance as ph5
import phase6_reference_standard_sensitivity as ph6

# Merkez anonimlestirme haritasi -- manuscript metni "Centre 1/2/3" kullaniyor, ham
# veri kolonlari ise gercek merkez kodlarini (RUMC/ZGT/PCNN) tasiyor. Tablo/figur
# ciktilarinda GORUNEN her yerde bu harita uygulanmali, aksi halde kod ciktisi
# (CSV/MD) manuscript metniyle celisir ve ham merkez kodlarini disariya sizdirir.
CENTRE_DISPLAY_NAMES = {'RUMC': 'Centre 1', 'ZGT': 'Centre 2', 'PCNN': 'Centre 3'}

FIGURE_DPI = 600
# NOT: Dergi formati kilavuzu "bitmapped line drawings" icin min 1000dpi
# istiyor; kullanici acikca 600dpi TIFF istedi. Cizgi-grafik (ROC/flow/
# calibration/DCA) oldugu icin submission oncesi bu deger dergiyle teyit
# edilmeli -- gerekirse FIGURE_DPI=1000 yapmak yeterli, kod tarafinda tek
# satir degisiklik.


def print_header(title):
    print("\n" + "=" * 90)
    print(f"||  {title}")
    print("=" * 90)


def print_subheader(title):
    print(f"\n{'-' * 80}\n  {title}\n{'-' * 80}")


# ============================================================
# ORTAK YARDIMCILAR
# ============================================================
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
    return csv_path, md_path


def save_figure(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, f"{name}.tiff")
    fig.savefig(path, format='tiff', dpi=FIGURE_DPI,
                pil_kwargs={'compression': 'tiff_lzw'}, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figur kaydedildi: {path} ({FIGURE_DPI} dpi, TIFF/LZW)")
    return path


def fmt_ci(lo, hi, dec=4):
    if lo is None or hi is None or (isinstance(lo, float) and np.isnan(lo)):
        return "—"
    return f"[{lo:.{dec}f}, {hi:.{dec}f}]"


def fmt_num(x, dec=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{dec}f}"


def median_iqr(series, decimals=1):
    s = pd.Series(series).dropna()
    if len(s) == 0:
        return "—", 0
    med = s.median()
    q1, q3 = s.quantile([0.25, 0.75])
    return f"{med:.{decimals}f} ({q1:.{decimals}f}-{q3:.{decimals}f})", int(pd.Series(series).isna().sum())


def concordance_correlation_coefficient(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    sxy = np.mean((x - mx) * (y - my))
    return 2 * sxy / (vx + vy + (mx - my) ** 2)


_SUPERSCRIPT_MAP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def fmt_p(p):
    """Manuscript-style p-value: '<0.001' below threshold, else 3 decimals."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "—"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fmt_sci_p(p):
    """Manuscript-style scientific-notation p-value, e.g. 7.487e-10 -> '7.5x10^-10'
    (unicode superscript), used for very small LRT p-values."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "—"
    if p >= 0.001:
        return fmt_p(p)
    exponent = int(np.floor(np.log10(abs(p))))
    mantissa = p / (10 ** exponent)
    return f"{mantissa:.1f}×10{str(exponent).translate(_SUPERSCRIPT_MAP)}"


# ============================================================
# VERI YUKLEME (Phase 1-5 ile BIREBIR AYNI kohort tanimlari,
# ph3/ph4/ph5 modullerinden dogrudan cagriliyor)
# ============================================================
def load_data():
    df = pd.read_csv(MERGED_CSV_PATH)
    cohort_a = ph3.build_cohort_a(df)          # N=1418, calculated_manual_psad/ai_psad/reported_psad dahil
    cohort_b = ph3.build_cohort_b(df)           # N=1096, log_PSA/log_manual_volume/log_AI_volume/centre_RUMC dahil
    return df, cohort_a, cohort_b


# ============================================================
# TABLE 1 - BASELINE CHARACTERISTICS (Cohort A + Cohort B footnote)
# ============================================================
def build_table1(cohort_a, cohort_b):
    print_header("TABLE 1 - Baseline characteristics")
    pos = cohort_a[cohort_a['csPCa_binary'] == 1]
    neg = cohort_a[cohort_a['csPCa_binary'] == 0]
    n = len(cohort_a)

    vars_to_report = [
        ('patient_age', 'Age, years', 1),
        ('psa', 'PSA, ng/mL', 1),
        ('prostate_volume', 'Clinically reported prostate volume, mL', 1),
        ('ai_volume_Bosma22b', 'AI-derived prostate volume, mL', 1),
        ('calculated_manual_psad', 'Calculated manual PSAD, ng/mL/mL', 2),
        ('ai_psad', 'AI-PSAD, ng/mL/mL', 2),
    ]
    rows = []
    for col, label, dec in vars_to_report:
        overall, n_missing = median_iqr(cohort_a[col], dec)
        pos_str, _ = median_iqr(pos[col], dec)
        neg_str, _ = median_iqr(neg[col], dec)
        valid_pos, valid_neg = pos[col].dropna(), neg[col].dropna()
        if len(valid_pos) > 0 and len(valid_neg) > 0:
            _, p = stats.mannwhitneyu(valid_pos, valid_neg, alternative='two-sided')
            p_str = "<0.001" if p < 0.001 else f"{p:.3f}"
        else:
            p_str = "N/A"
        rows.append({'Variable': label, 'Overall': overall, 'csPCa_positive': pos_str,
                     'csPCa_negative': neg_str, 'p_value': p_str, 'n_missing': n_missing})

    n_reported = cohort_a['reported_psad'].notna().sum()
    rows.append({'Variable': 'Reported PSAD field available, n (%)',
                 'Overall': f"{n_reported} ({100 * n_reported / n:.1f})",
                 'csPCa_positive': f"{pos['reported_psad'].notna().sum()} "
                                   f"({100 * pos['reported_psad'].notna().mean():.1f})",
                 'csPCa_negative': f"{neg['reported_psad'].notna().sum()} "
                                   f"({100 * neg['reported_psad'].notna().mean():.1f})",
                 'p_value': '—', 'n_missing': 0})

    for c in ['RUMC', 'ZGT', 'PCNN']:
        n_c = int((cohort_a['center'] == c).sum())
        n_c_pos = int(((cohort_a['center'] == c) & (cohort_a['csPCa_binary'] == 1)).sum())
        n_c_neg = n_c - n_c_pos
        rows.append({'Variable': f'Centre — {CENTRE_DISPLAY_NAMES[c]}, n', 'Overall': n_c,
                     'csPCa_positive': n_c_pos, 'csPCa_negative': n_c_neg,
                     'p_value': '—', 'n_missing': 0})

    for grade in sorted(cohort_a['case_ISUP'].dropna().unique()):
        label = f"ISUP grade group {int(grade)}" if grade > 0 else "Negative (benign)"
        n_g = int((cohort_a['case_ISUP'] == grade).sum())
        n_g_pos = int(((cohort_a['case_ISUP'] == grade) & (cohort_a['csPCa_binary'] == 1)).sum())
        n_g_neg = n_g - n_g_pos
        rows.append({'Variable': label, 'Overall': n_g,
                     'csPCa_positive': n_g_pos, 'csPCa_negative': n_g_neg,
                     'p_value': '—', 'n_missing': 0})

    table1_df = pd.DataFrame(rows)

    n_b = len(cohort_b)
    events_b = int(cohort_b['csPCa_binary'].sum())
    centre_b_lines = []
    for c in sorted(cohort_b['center'].unique()):
        sub = cohort_b[cohort_b['center'] == c]
        centre_b_lines.append(f"{CENTRE_DISPLAY_NAMES[c]} n={len(sub)} ({100 * len(sub) / n_b:.1f}%), "
                               f"csPCa+={int(sub['csPCa_binary'].sum())} "
                               f"({100 * sub['csPCa_binary'].mean():.1f}%)")
    footnote = (
        f"Note. Cohort A (n={n}) is the primary single-marker analytic cohort (calculated manual "
        f"PSAD vs AI-PSAD); age and centre were not required because they are not covariates in this "
        f"comparison. Cohort B (n={n_b}, csPCa+ {events_b} [{100*events_b/n_b:.1f}%]) is the "
        f"multivariable model cohort (Model A/A'), restricted to Centre 1 and Centre 2 (Centre 3 "
        f"excluded by prespecified design to preserve the two-centre validation framework): "
        + "; ".join(centre_b_lines) + ". "
        f"csPCa was defined as ISUP grade group >=2. Reported PSAD was not used as an inclusion "
        f"criterion and is retained only as a secondary comparator (Table S1)."
    )
    save_table(table1_df, "Table1_baseline_characteristics", notes=footnote)
    return table1_df


# ============================================================
# TABLE 2 - DISCRIMINATION + VALIDATION + CALIBRATION SUMMARY
# ============================================================
def build_table2(df, cohort_a, cohort_b):
    print_header("TABLE 2 - Discrimination, validation, and calibration summary")
    rows = []

    # -- Cohort A single-marker (3.3) --
    sm_result = ph3.run_primary_single_marker(cohort_a)
    rows.append({'analysis': 'Calculated manual PSAD', 'cohort': 'Cohort A', 'n': sm_result['n'],
                 'events': sm_result['events'], 'model_or_marker': 'Calculated manual PSAD',
                 'auc_or_delta': sm_result['auc_calculated_manual_psad'],
                 'ci_lo': sm_result['ci_calc'][0], 'ci_hi': sm_result['ci_calc'][1],
                 'p_type': 'DeLong CI', 'p_value': np.nan, 'brier': np.nan,
                 'calib_intercept': np.nan, 'calib_slope': np.nan, 'n_boot': np.nan, 'n_boot_failed': np.nan})
    rows.append({'analysis': 'AI-PSAD', 'cohort': 'Cohort A', 'n': sm_result['n'],
                 'events': sm_result['events'], 'model_or_marker': 'AI-PSAD',
                 'auc_or_delta': sm_result['auc_ai_psad'],
                 'ci_lo': sm_result['ci_ai'][0], 'ci_hi': sm_result['ci_ai'][1],
                 'p_type': 'DeLong CI', 'p_value': np.nan, 'brier': np.nan,
                 'calib_intercept': np.nan, 'calib_slope': np.nan, 'n_boot': np.nan, 'n_boot_failed': np.nan})
    boot = sm_result['bootstrap_delta']
    rows.append({'analysis': 'Delta (AI-PSAD - calculated manual PSAD)', 'cohort': 'Cohort A',
                 'n': sm_result['n'], 'events': sm_result['events'], 'model_or_marker': 'AI-PSAD vs calculated',
                 'auc_or_delta': boot['delta_median'], 'ci_lo': boot['delta_ci_lo'], 'ci_hi': boot['delta_ci_hi'],
                 'p_type': 'DeLong p (supportive)', 'p_value': sm_result['delong_p'], 'brier': np.nan,
                 'calib_intercept': np.nan, 'calib_slope': np.nan, 'n_boot': boot['n_boot'],
                 'n_boot_failed': boot['n_failed']})

    # -- Cohort B Model A / A' apparent fit + combined bootstrap (3.1/3.4/3.5) --
    y_b = cohort_b['csPCa_binary'].values
    age, psa = cohort_b['age_per_10'].values, cohort_b['log_PSA'].values
    vol_m, vol_ai, centre = (cohort_b['log_manual_volume'].values, cohort_b['log_AI_volume'].values,
                             cohort_b['centre_RUMC'].values)
    X_a = sm.add_constant(np.column_stack([age, psa, vol_m, centre]))
    X_ap = sm.add_constant(np.column_stack([age, psa, vol_ai, centre]))
    var_names = ['Intercept', 'age_per_10', 'log_PSA', 'log_volume', 'centre_RUMC']
    apparent_a = ph3.fit_and_report_model(y_b, X_a, var_names, "Model A (clinically reported volume)")
    apparent_ap = ph3.fit_and_report_model(y_b, X_ap, var_names, "Model A' (AI-derived volume)")
    _, _, p_delong_models = ph3.delong_paired_test(y_b, apparent_a['pred'], apparent_ap['pred'])
    combined = ph3.combined_bootstrap_ratio_and_calibration(cohort_b, apparent_a, apparent_ap)

    for label, res, calib in [('Model A (clinically reported volume)', apparent_a, combined['model_a_calibration']),
                               ("Model A' (AI-derived volume)", apparent_ap, combined['model_ap_calibration'])]:
        rows.append({'analysis': f'{label}, apparent', 'cohort': 'Cohort B', 'n': res['n'], 'events': res['events'],
                     'model_or_marker': label, 'auc_or_delta': res['auc'], 'ci_lo': res['auc_lo'],
                     'ci_hi': res['auc_hi'], 'p_type': 'DeLong CI', 'p_value': np.nan,
                     'brier': calib['apparent_brier'], 'calib_intercept': calib['apparent_intercept'],
                     'calib_slope': calib['apparent_slope'], 'n_boot': np.nan, 'n_boot_failed': np.nan})
        rows.append({'analysis': f'{label}, optimism-corrected', 'cohort': 'Cohort B', 'n': res['n'],
                     'events': res['events'], 'model_or_marker': label, 'auc_or_delta': calib['corrected_auc'],
                     'ci_lo': np.nan, 'ci_hi': np.nan, 'p_type': 'Bootstrap optimism-correction',
                     'p_value': np.nan, 'brier': calib['corrected_brier'],
                     'calib_intercept': calib['corrected_intercept'], 'calib_slope': calib['corrected_slope'],
                     'n_boot': combined['n_boot'], 'n_boot_failed': combined['n_failed']})

    delta_primary = combined['delta_auc_bootstrap_primary']
    rows.append({'analysis': "Delta (Model A' - Model A), apparent", 'cohort': 'Cohort B', 'n': len(cohort_b),
                 'events': int(y_b.sum()), 'model_or_marker': "Model A' vs A",
                 'auc_or_delta': delta_primary['median'], 'ci_lo': delta_primary['ci_lo'],
                 'ci_hi': delta_primary['ci_hi'], 'p_type': 'DeLong p (supportive)', 'p_value': p_delong_models,
                 'brier': np.nan, 'calib_intercept': np.nan, 'calib_slope': np.nan,
                 'n_boot': combined['n_boot'], 'n_boot_failed': combined['n_failed']})

    # -- Centre-based validation, brief (4.1) --
    for train_c, test_c, lbl in [('RUMC', 'ZGT', 'Centre 1->Centre 2'), ('ZGT', 'RUMC', 'Centre 2->Centre 1')]:
        r = ph4.run_center_validation_direction(cohort_b, train_c, test_c, lbl)
        if r is None:
            continue
        d = r['delta_auc_bootstrap']
        rows.append({'analysis': f"Centre-based validation, {lbl} (Model A' vs A, test set)",
                     'cohort': f"Cohort B, train={CENTRE_DISPLAY_NAMES[train_c]} "
                               f"test={CENTRE_DISPLAY_NAMES[test_c]}", 'n': r['n_test'],
                     'events': r['events_test'], 'model_or_marker': "Model A' vs A",
                     'auc_or_delta': d['median'], 'ci_lo': d['ci_lo'], 'ci_hi': d['ci_hi'],
                     'p_type': 'Bootstrap CI', 'p_value': np.nan,
                     'brier': r['brier_ap'], 'calib_intercept': r['calib_ap'][0], 'calib_slope': r['calib_ap'][1],
                     'n_boot': d['n_boot'], 'n_boot_failed': d['n_failed']})

    table2_raw = pd.DataFrame(rows)

    display_rows = []
    for _, r in table2_raw.iterrows():
        display_rows.append({
            'Analysis': r['analysis'], 'Cohort (n/events)': f"{r['cohort']} ({int(r['n'])}/{int(r['events'])})",
            'AUC or ΔAUC': fmt_num(r['auc_or_delta']),
            'CI': fmt_ci(r['ci_lo'], r['ci_hi']),
            'p (type)': (f"{r['p_value']:.4g} ({r['p_type']})" if not pd.isna(r['p_value']) else r['p_type']),
            'Brier': fmt_num(r['brier'], 3), 'Calib. intercept': fmt_num(r['calib_intercept'], 3),
            'Calib. slope': fmt_num(r['calib_slope'], 3),
        })
    table2_display = pd.DataFrame(display_rows)

    n_boot_failed_total = int(table2_raw['n_boot_failed'].fillna(0).sum())
    footnote = (
        f"Note. AUC values apparent unless labelled optimism-corrected. Delta-AUC 95% CIs use paired "
        f"patient-level bootstrap resampling of the statistic (resample-the-statistic method; a "
        f"fixed-cohort/optimism-style resampling was used only for bias correction, not for CIs, per "
        f"Table 3 methodology). Repeated cross-validation and full centre-validation/ProstateX/3-centre "
        f"sensitivity detail are in Supplementary Table S3; PSAD threshold, reclassification, and "
        f"decision-curve results are in Supplementary Table S2. No bootstrap resamples failed to "
        f"converge across the analyses summarized here (total nonconvergent resamples = "
        f"{n_boot_failed_total})."
    )
    save_table(table2_raw, "Table2_discrimination_validation_calibration_RAW", notes=None)
    save_table(table2_display, "Table2_discrimination_validation_calibration", notes=footnote)
    return table2_raw, table2_display, dict(apparent_a=apparent_a, apparent_ap=apparent_ap, combined=combined)


# ============================================================
# TABLE 3 - COEFFICIENTS + LRT + COEFFICIENT RATIO
# ============================================================
def build_table3(cohort_b, table2_context):
    print_header("TABLE 3 - Model coefficients, LRT, coefficient ratio")
    apparent_a, apparent_ap, combined = (table2_context['apparent_a'], table2_context['apparent_ap'],
                                          table2_context['combined'])
    y_b = cohort_b['csPCa_binary'].values
    age, psa = cohort_b['age_per_10'].values, cohort_b['log_PSA'].values
    vol_m, vol_ai, centre = (cohort_b['log_manual_volume'].values, cohort_b['log_AI_volume'].values,
                             cohort_b['centre_RUMC'].values)
    lrt_a = ph3.run_lrt(y_b, age, vol_m, psa, centre, "Model A (clinically reported volume)")
    lrt_ap = ph3.run_lrt(y_b, age, vol_ai, psa, centre, "Model A' (AI-derived volume)")

    # -- RAW/long format (bir model+degisken per satir) -- tam reprodusibilite icin korunuyor --
    raw_rows = []
    for label, res in [('Model A', apparent_a), ("Model A'", apparent_ap)]:
        for i, vname in enumerate(res['var_names']):
            coef, se, p = res['params'][i], res['ses'][i], res['pvals'][i]
            or_str, ci_str = "—", "—"
            if vname != 'Intercept':
                or_val = np.exp(coef)
                ci_lo, ci_hi = np.exp(coef - 1.96 * se), np.exp(coef + 1.96 * se)
                or_str, ci_str = f"{or_val:.3f}", f"({ci_lo:.3f}-{ci_hi:.3f})"
            raw_rows.append({'model': label, 'variable': vname, 'coef': coef, 'se': se,
                              'OR': or_str, '95%_CI': ci_str, 'p': p})
        raw_rows.append({'model': label, 'variable': 'AIC', 'coef': res['aic'], 'se': np.nan,
                          'OR': '—', '95%_CI': '—', 'p': np.nan})
        raw_rows.append({'model': label, 'variable': 'pseudo-R2 (McFadden)', 'coef': res['pseudo_r2'],
                          'se': np.nan, 'OR': '—', '95%_CI': '—', 'p': np.nan})
    coef_raw_df = pd.DataFrame(raw_rows)

    ratio_raw_rows = [
        {'model': 'Model A', 'ratio_point_estimate': combined['ratio_a']['point_estimate'],
         'bootstrap_median': combined['ratio_a']['median'], 'ci_lo': combined['ratio_a']['ci_lo'],
         'ci_hi': combined['ratio_a']['ci_hi'], 'lrt_chi2': lrt_a['lr_chi2'], 'lrt_df': lrt_a['df'],
         'lrt_p': lrt_a['p_value'], 'restricted_composite_coef': lrt_a['composite_coef']},
        {'model': "Model A'", 'ratio_point_estimate': combined['ratio_ap']['point_estimate'],
         'bootstrap_median': combined['ratio_ap']['median'], 'ci_lo': combined['ratio_ap']['ci_lo'],
         'ci_hi': combined['ratio_ap']['ci_hi'], 'lrt_chi2': lrt_ap['lr_chi2'], 'lrt_df': lrt_ap['df'],
         'lrt_p': lrt_ap['p_value'], 'restricted_composite_coef': lrt_ap['composite_coef']},
    ]
    ratio_raw_df = pd.DataFrame(ratio_raw_rows)

    # -- DISPLAY/wide format -- manuscript-hazir, iki panelli, sade tablo --
    def coef_cell(res, vname):
        i = res['var_names'].index(vname)
        coef, se = res['params'][i], res['ses'][i]
        if vname == 'Intercept':
            return f"{coef:.3f}"
        or_val = np.exp(coef)
        ci_lo, ci_hi = np.exp(coef - 1.96 * se), np.exp(coef + 1.96 * se)
        return f"{coef:.3f} (OR {or_val:.2f}, {ci_lo:.2f}–{ci_hi:.2f})"

    panelA_rows = []
    for vname, label in [('Intercept', 'Intercept'), ('age_per_10', 'Age (per 10 years)'),
                         ('log_PSA', 'PSA (log)'), ('log_volume', 'Volume (log)'),
                         ('centre_RUMC', 'Centre 1 (vs Centre 2)')]:
        i_a, i_ap = apparent_a['var_names'].index(vname), apparent_ap['var_names'].index(vname)
        panelA_rows.append({
            'Variable': label,
            'Model A: Coef (OR, 95% CI)': coef_cell(apparent_a, vname),
            'p (Model A)': fmt_p(apparent_a['pvals'][i_a]),
            "Model A': Coef (OR, 95% CI)": coef_cell(apparent_ap, vname),
            "p (Model A')": fmt_p(apparent_ap['pvals'][i_ap]),
        })
    panelA_rows.append({'Variable': 'AIC', 'Model A: Coef (OR, 95% CI)': f"{apparent_a['aic']:.1f}",
                        'p (Model A)': '—', "Model A': Coef (OR, 95% CI)": f"{apparent_ap['aic']:.1f}",
                        "p (Model A')": '—'})
    panelA_rows.append({'Variable': 'Pseudo-R² (McFadden)',
                        'Model A: Coef (OR, 95% CI)': f"{apparent_a['pseudo_r2']:.3f}", 'p (Model A)': '—',
                        "Model A': Coef (OR, 95% CI)": f"{apparent_ap['pseudo_r2']:.3f}", "p (Model A')": '—'})
    coef_display_df = pd.DataFrame(panelA_rows)

    panelB_rows = [
        {'Model': 'Model A',
         'Volume-to-PSA ratio (95% CI)': f"{combined['ratio_a']['point_estimate']:.2f} "
                                          f"({combined['ratio_a']['ci_lo']:.2f}–{combined['ratio_a']['ci_hi']:.2f})",
         'LRT chi2 (df=1)': f"{lrt_a['lr_chi2']:.2f}", 'LRT p': fmt_sci_p(lrt_a['p_value'])},
        {'Model': "Model A'",
         'Volume-to-PSA ratio (95% CI)': f"{combined['ratio_ap']['point_estimate']:.2f} "
                                          f"({combined['ratio_ap']['ci_lo']:.2f}–{combined['ratio_ap']['ci_hi']:.2f})",
         'LRT chi2 (df=1)': f"{lrt_ap['lr_chi2']:.2f}", 'LRT p': fmt_sci_p(lrt_ap['p_value'])},
    ]
    ratio_display_df = pd.DataFrame(panelB_rows)

    footnote = (
        "Note. Model A used clinically reported prostate volume; Model A' used AI-derived prostate "
        "volume. Odds ratios and coefficient ratios with 95% CIs are dataset-specific, model-based "
        "estimates, not universal clinical weighting factors. The likelihood-ratio test (LRT) compares "
        "each freely estimated model against a restricted model constrained to the conventional PSAD "
        "equal-and-opposite weighting (a single composite log[PSA/volume] predictor); rejection of this "
        "constraint (p<0.001 for both models) indicates PSA and volume were not equally weighted in "
        "this dataset. Functional-form and collinearity/VIF diagnostics are reported in Supplementary "
        "Table S3."
    )
    save_table(coef_raw_df, "Table3_coefficients_RAW")
    save_table(ratio_raw_df, "Table3_coefficient_ratio_and_LRT_RAW")
    save_table(coef_display_df, "Table3_coefficients")
    save_table(ratio_display_df, "Table3_coefficient_ratio_and_LRT", notes=footnote)
    return coef_display_df, ratio_display_df, dict(lrt_a=lrt_a, lrt_ap=lrt_ap)


# ============================================================
# FIGURE 1 - PARTICIPANT FLOW DIAGRAM
# ============================================================
# Toplam PI-CAI Public Training and Development Dataset exam sayisi (1500),
# dataset dokumantasyonunun sabit bir gercegi (Saha et al., PMID 38876123) --
# yerel bir CSV'den turetilebilecek bir "sonuc" degil, bu yuzden burada
# sabit olarak yaziliyor (analiz sonucu degil, kaynak-belge referans sayisi).
TOTAL_EXAMS_PUBLIC_DATASET = 1500


def build_figure1(df, cohort_a, cohort_b):
    print_header("FIGURE 1 - Participant flow diagram")
    n_patients = len(df)
    n_cohort_a = len(cohort_a)
    n_cohort_b = len(cohort_b)
    n_reported_subset = int(cohort_a['reported_psad'].notna().sum())

    try:
        prostatex_result = ph4.run_prostatex_sensitivity(cohort_b)
        n_nonprostatex = prostatex_result['n_remaining'] if prostatex_result else None
    except Exception as e:
        print(f"  UYARI: ProstateX sensitivity Figure 1 icin hesaplanamadi ({e}).")
        n_nonprostatex = None

    n_3centre = len(ph4.build_cohort_b_3centre(pd.read_csv(MERGED_CSV_PATH)))

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    def box(x, y, w, h, text, fontsize=8.5):
        rect = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05",
                               edgecolor='black', facecolor='white', linewidth=1.1)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle='-|>', linewidth=1.1, color='black'))

    box(5, 9.4, 8.6, 0.8, f"PI-CAI Public Training and Development Dataset\n"
                          f"{TOTAL_EXAMS_PUBLIC_DATASET} examinations")
    arrow(5, 9.0, 5, 8.4)
    box(5, 8.0, 8.6, 0.8, f"Patient-level deduplication; known-faulty/implausible\n"
                          f"AI-volume exclusion -> {n_patients} unique patients")
    arrow(5, 7.6, 5, 7.0)
    box(5, 6.6, 8.6, 0.8, f"Cohort A (single-marker PSAD comparison), n={n_cohort_a}\n"
                          f"(PSA, clinically reported volume, AI-volume, csPCa outcome available)")

    # NOT: bu satirdaki 3 kutu daha once genislikleri toplami (3.6+4.2+3.6=11.4) tuval
    # genisligini (10 birim) astigi icin ust uste biniyordu -- asagida daraltilip
    # araliklandirildi (0.25-3.15 / 3.35-6.65 / 6.85-9.75, aralarinda bosluk var).
    arrow(3.6, 6.2, 1.7, 5.4)
    box(1.7, 5.0, 2.9, 0.9, f"Reported-PSAD secondary\ncomparator subset\nn={n_reported_subset}", fontsize=7.3)

    arrow(6.4, 6.2, 8.3, 5.4)
    box(8.3, 5.0, 2.9, 0.9, f"Three-centre exploratory\nsensitivity (PCNN included)\nn={n_3centre}", fontsize=7.3)

    arrow(5, 6.2, 5, 5.4)
    box(5, 5.0, 3.3, 0.9, f"Cohort B (multivariable model,\nModel A / Model A'), n={n_cohort_b}\n"
                          f"RUMC + ZGT only", fontsize=7.3)

    arrow(5, 4.55, 3.4, 3.7)
    nptx_txt = f"n={n_nonprostatex}" if n_nonprostatex is not None else "n=[TO INSERT]"
    box(3.0, 3.3, 3.6, 0.9, f"Non-ProstateX sensitivity\n{nptx_txt}", fontsize=8)

    arrow(5, 4.55, 6.6, 3.7)
    box(7.0, 3.3, 3.6, 0.9, "Centre-based validation\n(RUMC<->ZGT train/test splits\nwithin Cohort B)", fontsize=8)

    # NOT: baslik gomulmuyor -- dergi kurali "caption baslikta yer almaz, ayri verilir" diyor.
    path = save_figure(fig, "Figure1_flow_diagram")

    flow_df = pd.DataFrame([
        {'node': 'PI-CAI public dataset (examinations)', 'n': TOTAL_EXAMS_PUBLIC_DATASET},
        {'node': 'Unique patients after dedup/QC', 'n': n_patients},
        {'node': 'Cohort A (single-marker)', 'n': n_cohort_a},
        {'node': 'Reported-PSAD secondary subset', 'n': n_reported_subset},
        {'node': 'Three-centre exploratory sensitivity', 'n': n_3centre},
        {'node': 'Cohort B (multivariable model)', 'n': n_cohort_b},
        {'node': 'Non-ProstateX sensitivity', 'n': n_nonprostatex if n_nonprostatex is not None else np.nan},
    ])
    save_table(flow_df, "Figure1_flow_diagram_node_counts")
    return path


# ============================================================
# FIGURE 2 - ROC CURVES (Cohort A markers; Cohort B models)
# ============================================================
def build_figure2(cohort_a, cohort_b, table2_context):
    print_header("FIGURE 2 - ROC curves")
    apparent_a, apparent_ap = table2_context['apparent_a'], table2_context['apparent_ap']

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))

    y_a = cohort_a['csPCa_binary'].values
    for score_col, label in [('calculated_manual_psad', 'Calculated manual PSAD'), ('ai_psad', 'AI-PSAD')]:
        score = cohort_a[score_col].values
        fpr, tpr, _ = roc_curve(y_a, score)
        auc_val = roc_auc_score(y_a, score)
        axes[0].plot(fpr, tpr, linewidth=1.8, label=f"{label} (AUC={auc_val:.3f})")
    axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
    axes[0].set_xlabel('1 - Specificity')
    axes[0].set_ylabel('Sensitivity')
    axes[0].set_title(f"A. Cohort A single-marker comparison\n(n={len(cohort_a)}, "
                       f"events={int(y_a.sum())})", fontsize=9.5)
    axes[0].legend(loc='lower right', fontsize=8)

    y_b = cohort_b['csPCa_binary'].values
    for res, label in [(apparent_a, 'Model A (clinically reported volume)'),
                       (apparent_ap, "Model A' (AI-derived volume)")]:
        fpr, tpr, _ = roc_curve(y_b, res['pred'])
        axes[1].plot(fpr, tpr, linewidth=1.8, label=f"{label} (AUC={res['auc']:.3f})")
    axes[1].plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
    axes[1].set_xlabel('1 - Specificity')
    axes[1].set_ylabel('Sensitivity')
    axes[1].set_title(f"B. Cohort B multivariable models\n(n={apparent_a['n']}, "
                       f"events={apparent_a['events']})", fontsize=9.5)
    axes[1].legend(loc='lower right', fontsize=8)

    # NOT: baslik gomulmuyor (dergi kurali) -- panel A/B alt-basliklari (n/events ile) kaliyor.
    plt.tight_layout()
    return save_figure(fig, "Figure2_ROC_curves")


# ============================================================
# TABLE S1 - COHORT DERIVATION, CENTRE DISTRIBUTION, REPORTED-PSAD AUDIT
# ============================================================
def build_table_s1(df, cohort_a, cohort_b):
    print_header("TABLE S1 - Cohort derivation, centre distribution, reported-PSAD audit")

    deriv_rows = [
        {'cohort': 'Cohort A (single-marker PSAD)', 'n': len(cohort_a),
         'cspca_pos': int(cohort_a['csPCa_binary'].sum()),
         'cspca_pct': round(100 * cohort_a['csPCa_binary'].mean(), 1)},
        {'cohort': 'Cohort B (multivariable model)', 'n': len(cohort_b),
         'cspca_pos': int(cohort_b['csPCa_binary'].sum()),
         'cspca_pct': round(100 * cohort_b['csPCa_binary'].mean(), 1)},
    ]
    for c in ['RUMC', 'ZGT', 'PCNN']:
        sub = cohort_a[cohort_a['center'] == c]
        n_rep = int(sub['reported_psad'].notna().sum())
        deriv_rows.append({'cohort': f'Reported-PSAD availability, Cohort A / {CENTRE_DISPLAY_NAMES[c]}',
                           'n': len(sub), 'cspca_pos': n_rep,
                           'cspca_pct': round(100 * n_rep / len(sub), 1) if len(sub) > 0 else np.nan})
    deriv_df = pd.DataFrame(deriv_rows)

    # Centre 3 eligibility audit: recompute dynamically (no hardcoded "322")
    cohort_b_3centre = ph4.build_cohort_b_3centre(df)
    n_centre3_eligible = int((cohort_b_3centre['center'] == 'PCNN').sum())
    n_centre3_cohort_a = int((cohort_a['center'] == 'PCNN').sum())
    audit_df = pd.DataFrame([{
        'n_centre3_eligible_for_cohort_b_criteria': n_centre3_eligible,
        'n_centre3_in_cohort_a': n_centre3_cohort_a,
        'identical': n_centre3_eligible == n_centre3_cohort_a,
        'interpretation': ("Centre 3 exclusion from Cohort B is a prespecified design choice, not a "
                            "missing-data limitation" if n_centre3_eligible == n_centre3_cohort_a else
                            "Centre 3 exclusion partly reflects missing data, not design alone")
    }])

    # Discordance analysis (reported vs calculated PSAD), recomputed directly
    sub = cohort_a.dropna(subset=['reported_psad']).copy()
    r, p_r = stats.pearsonr(sub['reported_psad'], sub['calculated_manual_psad'])
    rho, p_rho = stats.spearmanr(sub['reported_psad'], sub['calculated_manual_psad'])
    abs_diff = (sub['reported_psad'] - sub['calculated_manual_psad']).abs()
    ratio = sub['reported_psad'] / sub['calculated_manual_psad']
    fold3_mask = (ratio > 3) | (ratio < 1 / 3)
    sub_excl = sub[~fold3_mask]
    r_after, _ = stats.pearsonr(sub_excl['reported_psad'], sub_excl['calculated_manual_psad'])
    rho_after, _ = stats.spearmanr(sub_excl['reported_psad'], sub_excl['calculated_manual_psad'])
    abs_diff_after = (sub_excl['reported_psad'] - sub_excl['calculated_manual_psad']).abs()

    discordance_df = pd.DataFrame([
        {'metric': 'Pearson r', 'before_exclusion': r, 'after_excluding_3fold_outliers': r_after},
        {'metric': 'Spearman rho', 'before_exclusion': rho, 'after_excluding_3fold_outliers': rho_after},
        {'metric': 'Mean absolute difference', 'before_exclusion': abs_diff.mean(),
         'after_excluding_3fold_outliers': abs_diff_after.mean()},
        {'metric': 'Median absolute difference', 'before_exclusion': abs_diff.median(),
         'after_excluding_3fold_outliers': abs_diff_after.median()},
        {'metric': 'n', 'before_exclusion': len(sub), 'after_excluding_3fold_outliers': len(sub_excl)},
        {'metric': '>3-fold discordant cases excluded', 'before_exclusion': int(fold3_mask.sum()),
         'after_excluding_3fold_outliers': 0},
    ])

    # Panel E: reference-standard verification status by cohort (histology-verified vs
    # MRI-only negatives) -- 3.1 metni bu tabloya atif yapiyor, hardcode edilmeden
    # Phase 6'nin marksheet-join fonksiyonlari dogrudan cagirilarak uretiliyor.
    try:
        mk_histopath = ph6.load_marksheet_histopath()
        df_source = df.copy()
        df_source['csPCa_binary'] = (df_source['case_csPCa'] == 'YES').astype(int)
        df_source_verif = ph6.join_verification_status(df_source, mk_histopath)
        cohort_a_verif = ph6.join_verification_status(cohort_a.copy(), mk_histopath)
        cohort_b_verif = ph6.join_verification_status(cohort_b.copy(), mk_histopath)
        verif_rows = [
            ph6.verification_composition(df_source_verif, "Source dataset"),
            ph6.verification_composition(cohort_a_verif, "Cohort A"),
            ph6.verification_composition(cohort_b_verif, "Cohort B"),
        ]
        verif_df = pd.DataFrame(verif_rows)[
            ['cohort', 'n_total', 'n_cspca_positive', 'n_cspca_negative',
             'n_negative_histology_verified', 'pct_negative_histology_verified',
             'n_negative_mri_only', 'pct_negative_mri_only',
             'n_histology_verified_sensitivity_cohort']
        ].rename(columns={
            'n_total': 'N', 'n_cspca_positive': 'csPCa+',
            'n_cspca_negative': 'csPCa-', 'n_negative_histology_verified': 'csPCa- histology-verified',
            'pct_negative_histology_verified': '% histology-verified (of negatives)',
            'n_negative_mri_only': 'csPCa- MRI-only (no biopsy)',
            'pct_negative_mri_only': '% MRI-only (of negatives)',
            'n_histology_verified_sensitivity_cohort': 'Histology-verified sensitivity cohort N',
        })
    except Exception as e:
        print(f"  UYARI: Table S1 Panel E (verification status) hesaplanamadi ({e}).")
        verif_df = pd.DataFrame([{'note': f'[TO INSERT — verification status computation failed: {e}]'}])

    save_table(deriv_df, "TableS1_partA_cohort_derivation_and_centre_distribution")
    save_table(audit_df, "TableS1_partB_centre3_eligibility_audit")
    save_table(discordance_df, "TableS1_partC_reported_psad_discordance",
               notes="Note. Discordance computed on the matched subset with both reported and "
                     "calculated manual PSAD available (subset of Cohort A).")
    save_table(verif_df, "TableS1_partE_reference_standard_verification_status",
               notes="Note. Histologically verified = histopath_type in {MRBx, SysBx, SysBx+MRBx, RP}. "
                     "MRI-only = csPCa-negative case with no biopsy performed. All csPCa-positive cases "
                     "had non-missing histopath_type. The histology-verified sensitivity cohort includes "
                     "all csPCa-positive cases plus only histologically verified csPCa-negative cases "
                     "(Section 3.5).")
    return deriv_df, audit_df, discordance_df, verif_df


# ============================================================
# TABLE S2 - THRESHOLD PERFORMANCE + RECLASSIFICATION + DCA
# ============================================================
def build_table_s2(cohort_a, cohort_b):
    print_header("TABLE S2 - Threshold performance, reclassification, decision-curve summary")

    psad_threshold_df = ph5.run_psad_threshold_analysis(cohort_a)
    reclass_df = ph5.run_reclassification(cohort_a, threshold=0.15)
    pred_a, pred_ap, valid = ph5.generate_averaged_oof_predictions(cohort_b)
    model_threshold_df = ph5.run_model_threshold_analysis(cohort_b, pred_a, pred_ap, valid)
    dca_df = ph5.run_dca(cohort_b, pred_a, pred_ap, valid)

    save_table(psad_threshold_df, "TableS2_partA_PSAD_threshold_performance")
    save_table(model_threshold_df, "TableS2_partB_model_threshold_performance")
    save_table(reclass_df, "TableS2_partC_reclassification_at_015")
    save_table(dca_df, "TableS2_partD_decision_curve_analysis",
               notes="Note. Based on repeated-CV averaged out-of-fold predictions (Cohort B), not "
                     "apparent/in-sample predictions. Bootstrap 95% CIs crossing zero indicate the net "
                     "benefit difference at that threshold was not statistically significant; treat as "
                     "supportive, not confirmatory, evidence of clinical utility.")
    return dict(psad_threshold=psad_threshold_df, model_threshold=model_threshold_df,
                reclass=reclass_df, dca=dca_df, pred_a=pred_a, pred_ap=pred_ap, valid=valid)


# ============================================================
# TABLE S3 - CENTRE VALIDATION + PROSTATEX + 3-CENTRE + FUNCTIONAL-FORM + EXPLORATORY
# ============================================================
def build_table_s3(df, cohort_a, cohort_b, table2_context):
    print_header("TABLE S3 - Validation, sensitivity, functional-form, exploratory analyses")

    # Panel A: full centre-validation detail
    panelA_rows = []
    for train_c, test_c, lbl in [('RUMC', 'ZGT', 'Centre 1->Centre 2'), ('ZGT', 'RUMC', 'Centre 2->Centre 1')]:
        r = ph4.run_center_validation_direction(cohort_b, train_c, test_c, lbl)
        if r is None:
            continue
        d = r['delta_auc_bootstrap']
        panelA_rows.append({
            'direction': lbl, 'n_train': r['n_train'], 'events_train': r['events_train'],
            'n_test': r['n_test'], 'events_test': r['events_test'],
            'auc_a': r['auc_a'], 'auc_a_ci_lo': r['auc_a_ci'][0], 'auc_a_ci_hi': r['auc_a_ci'][1],
            'brier_a': r['brier_a'], 'calib_intercept_a': r['calib_a'][0], 'calib_slope_a': r['calib_a'][1],
            'auc_ap': r['auc_ap'], 'auc_ap_ci_lo': r['auc_ap_ci'][0], 'auc_ap_ci_hi': r['auc_ap_ci'][1],
            'brier_ap': r['brier_ap'], 'calib_intercept_ap': r['calib_ap'][0], 'calib_slope_ap': r['calib_ap'][1],
            'delta_auc_median': d['median'], 'delta_ci_lo': d['ci_lo'], 'delta_ci_hi': d['ci_hi'],
            'n_boot': d['n_boot'], 'n_boot_failed': d['n_failed'],
            'ratio_a_train': r['ratio_a_train'], 'ratio_ap_train': r['ratio_ap_train'],
        })
    panelA_df = pd.DataFrame(panelA_rows)

    # Panel B: ProstateX sensitivity (full)
    prostatex_result = ph4.run_prostatex_sensitivity(cohort_b)
    panelB_df = pd.DataFrame([prostatex_result]) if prostatex_result else pd.DataFrame(
        [{'note': '[TO INSERT — ProstateX-mapping.json not found this run]'}])

    # Panel C: 3-centre sensitivity (full — duplicated minimal fit, ph4 fonksiyonu bunu donmuyor)
    apparent_a, apparent_ap = table2_context['apparent_a'], table2_context['apparent_ap']
    primary_delta_auc = apparent_ap['auc'] - apparent_a['auc']
    cohort_3c = ph4.build_cohort_b_3centre(df)
    y_3c = cohort_3c['csPCa_binary'].values
    age_3c, psa_3c = cohort_3c['age_per_10'].values, cohort_3c['log_PSA'].values
    vol_m_3c, vol_ai_3c = cohort_3c['log_manual_volume'].values, cohort_3c['log_AI_volume'].values
    centre_cols_3c = cohort_3c[['centre_RUMC', 'centre_PCNN']].values
    model_a_3c, model_ap_3c, X_a_3c, X_ap_3c = ph4.fit_model_a_and_ap(
        y_3c, age_3c, psa_3c, vol_m_3c, vol_ai_3c, centre_cols_3c)
    pred_a_3c, pred_ap_3c = model_a_3c.predict(X_a_3c), model_ap_3c.predict(X_ap_3c)
    auc_a_3c, lo_a_3c, hi_a_3c = ph4.delong_auc_ci(y_3c, pred_a_3c)
    auc_ap_3c, lo_ap_3c, hi_ap_3c = ph4.delong_auc_ci(y_3c, pred_ap_3c)
    result_3centre = ph4.run_3centre_sensitivity(df, primary_delta_auc)
    panelC_df = pd.DataFrame([{
        'n': result_3centre['n'], 'events': result_3centre['events'],
        'aic_model_a': model_a_3c.aic, 'auc_model_a': auc_a_3c, 'auc_a_ci_lo': lo_a_3c, 'auc_a_ci_hi': hi_a_3c,
        'aic_model_ap': model_ap_3c.aic, 'auc_model_ap': auc_ap_3c, 'auc_ap_ci_lo': lo_ap_3c, 'auc_ap_ci_hi': hi_ap_3c,
        'delong_p': result_3centre['delong_p'], 'delta_auc': result_3centre['delta_auc'],
        'primary_cohort_b_delta_auc': primary_delta_auc,
        'direction_consistent_with_primary': (result_3centre['delta_auc'] > 0) == (primary_delta_auc > 0),
    }])

    # Panel D: reference-standard sensitivity (histology-verified cohorts) -- Table 2's
    # footnote promises "reference-standard ... sensitivity analyses are reported in
    # Supplementary Table S3"; this panel is computed dynamically via Phase 6 functions
    # (no hardcoded numbers) so it can never silently drift from the real Phase 6 run.
    try:
        mk_histopath_s3 = ph6.load_marksheet_histopath()
        cohort_a_hv_input = ph6.join_verification_status(cohort_a.copy(), mk_histopath_s3)
        cohort_b_hv_input = ph6.join_verification_status(cohort_b.copy(), mk_histopath_s3)
        cohort_a_hv_result, _ = ph6.cohort_a_histology_sensitivity(cohort_a_hv_input)
        cohort_b_hv_result = ph6.cohort_b_histology_sensitivity(cohort_b_hv_input)
        a_row, b_row = cohort_a_hv_result.iloc[0], cohort_b_hv_result.iloc[0]

        panelD_refstd_df = pd.DataFrame([
            {'analysis': 'Single-marker PSAD', 'cohort': 'Histology-verified Cohort A',
             'n': int(a_row['n']), 'events': int(a_row['events']),
             'comparator': 'AI-PSAD vs calculated manual PSAD',
             'auc_1': a_row['auc_calculated_manual_psad'], 'auc_2': a_row['auc_ai_psad'],
             'delta_auc': a_row['bootstrap_delta_median'], 'delta_ci_lo': a_row['bootstrap_delta_ci_lo'],
             'delta_ci_hi': a_row['bootstrap_delta_ci_hi'], 'p_type': 'DeLong p (supportive)',
             'p_value': a_row['delong_p_supportive']},
            {'analysis': 'Multivariable model', 'cohort': 'Histology-verified Cohort B',
             'n': int(b_row['n']), 'events': int(b_row['events']),
             'comparator': "Model A' vs Model A",
             'auc_1': b_row['auc_model_a'], 'auc_2': b_row['auc_model_ap'],
             'delta_auc': b_row['bootstrap_delta_median'], 'delta_ci_lo': b_row['bootstrap_delta_ci_lo'],
             'delta_ci_hi': b_row['bootstrap_delta_ci_hi'], 'p_type': 'DeLong p (supportive)',
             'p_value': b_row['delong_p_supportive']},
            {'analysis': 'Constraint test, Model A', 'cohort': 'Histology-verified Cohort B',
             'n': int(b_row['n']), 'events': int(b_row['events']),
             'comparator': 'Free vs restricted PSAD-equivalent model',
             'auc_1': np.nan, 'auc_2': np.nan, 'delta_auc': np.nan, 'delta_ci_lo': np.nan, 'delta_ci_hi': np.nan,
             'p_type': 'LRT p', 'p_value': b_row['lrt_p_a']},
            {'analysis': "Constraint test, Model A'", 'cohort': 'Histology-verified Cohort B',
             'n': int(b_row['n']), 'events': int(b_row['events']),
             'comparator': 'Free vs restricted PSAD-equivalent model',
             'auc_1': np.nan, 'auc_2': np.nan, 'delta_auc': np.nan, 'delta_ci_lo': np.nan, 'delta_ci_hi': np.nan,
             'p_type': 'LRT p', 'p_value': b_row['lrt_p_ap']},
        ])
    except Exception as e:
        print(f"  UYARI: Table S3 Panel D (reference-standard sensitivity) hesaplanamadi ({e}).")
        panelD_refstd_df = pd.DataFrame([{'note': f'[TO INSERT — reference-standard sensitivity '
                                                   f'computation failed: {e}]'}])

    # Panel E: functional-form diagnostics (Phase 2 saved CSV'lerinden direkt okunuyor)
    panelE_frames = {}
    for fname in ['phase2_spline_functional_form_checks.csv', 'phase2_interaction_tests.csv',
                  'phase2_collinearity_correlations.csv', 'phase2_vif_diagnostics.csv']:
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(fpath):
            panelE_frames[fname] = pd.read_csv(fpath)
        else:
            print(f"  UYARI: {fpath} bulunamadi, Panel E bu parca icin [TO INSERT] birakiliyor.")
            panelE_frames[fname] = pd.DataFrame([{'note': f'[TO INSERT — {fname} not found]'}])

    # Panel F: exploratory ADC/lesion (bu revizyonun disinda, YENIDEN CALISTIRILMADI --
    # onceki kohort tanimini kullaniyor, bu acikca notta belirtiliyor)
    if os.path.exists(LEGACY_TRACK_A_CSV):
        panelF_track_a = pd.read_csv(LEGACY_TRACK_A_CSV)
    else:
        panelF_track_a = pd.DataFrame([{'note': '[TO INSERT — legacy faz3 Track A (whole-gland ADC) '
                                                 'output CSV not found this run; not re-derived, '
                                                 'out of scope of primary revision]'}])
    if os.path.exists(LEGACY_TRACK_B_CSV):
        panelF_track_b = pd.read_csv(LEGACY_TRACK_B_CSV)
    else:
        panelF_track_b = pd.DataFrame([{'note': '[TO INSERT — legacy faz3 Track B (lesion-level) '
                                                 'output CSV not found this run; not re-derived, '
                                                 'out of scope of primary revision]'}])

    save_table(panelA_df, "TableS3_partA_centre_validation_detail")
    save_table(panelB_df, "TableS3_partB_ProstateX_sensitivity_detail")
    save_table(panelC_df, "TableS3_partC_3centre_sensitivity_detail")
    save_table(panelD_refstd_df, "TableS3_partD_reference_standard_sensitivity_detail",
               notes="Note. Histology-verified sensitivity cohorts included all csPCa-positive cases "
                     "and only csPCa-negative cases with histological verification (MRI-targeted "
                     "biopsy, systematic biopsy, combined biopsy, or radical prostatectomy); MRI-only "
                     "negative cases without biopsy were excluded (Supplementary Table S1, Panel E).")
    for fname, fdf in panelE_frames.items():
        save_table(fdf, f"TableS3_partE_{fname.replace('.csv', '')}")
    save_table(panelF_track_a, "TableS3_partF_exploratory_ADC_heterogeneity",
               notes="Note. This exploratory analysis was not part of the primary revised cohort "
                     "framework (legacy cohort n=1098 vs current Cohort B n=1096) and was not "
                     "re-derived under the revised framework; reported as a hypothesis-generating "
                     "supplementary analysis only.")
    save_table(panelF_track_b, "TableS3_partF_exploratory_lesion_level",
               notes="Note. This exploratory analysis was not part of the primary revised cohort "
                     "framework and was not re-derived under the revised framework; reported as a "
                     "hypothesis-generating supplementary analysis only.")

    return dict(panelA=panelA_df, panelB=panelB_df, panelC=panelC_df, panelD=panelD_refstd_df,
                panelE=panelE_frames, panelF_track_a=panelF_track_a, panelF_track_b=panelF_track_b)


# ============================================================
# FIGURE S1 - BLAND-ALTMAN VOLUME AGREEMENT
# ============================================================
def build_figure_s1(df):
    print_header("FIGURE S1 - Bland-Altman volume agreement")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # NOT: kisa etiketler kullaniliyor -- eksen basliklari "Mean of X and Y, mL" gibi
    # uzun/tekrarli birlesik metinler yerine sabit/kisa tutuluyor, karsilastirilan
    # cift panel basliginda (kisa etiketlerle) belirtiliyor.
    pairs = [
        ('prostate_volume', 'ai_volume_Bosma22b', 'Clinical volume', 'AI volume (Alg. 1)', 0),
        ('ai_volume_Bosma22b', 'ai_volume_Guerbet23', 'AI volume (Alg. 1)',
         'AI volume (Alg. 2)', 1),
    ]
    agreement_rows = []
    for col_x, col_y, label_x, label_y, ax_idx in pairs:
        sub = df.dropna(subset=[col_x, col_y])
        x, y = sub[col_x].values, sub[col_y].values
        mean_vals = (x + y) / 2
        diff_vals = y - x
        bias = diff_vals.mean()
        sd_diff = diff_vals.std(ddof=1)
        loa_lo, loa_hi = bias - 1.96 * sd_diff, bias + 1.96 * sd_diff
        ccc = concordance_correlation_coefficient(x, y)
        pearson_r, _ = stats.pearsonr(x, y)

        ax = axes[ax_idx]
        ax.scatter(mean_vals, diff_vals, s=8, alpha=0.35, color='tab:blue')
        ax.axhline(bias, color='black', linewidth=1.2)
        ax.axhline(loa_lo, color='red', linestyle='--', linewidth=1)
        ax.axhline(loa_hi, color='red', linestyle='--', linewidth=1)
        ax.set_xlabel("Mean volume, mL")
        ax.set_ylabel("Difference, mL")
        ax.set_title(f"{'A' if ax_idx == 0 else 'B'}. {label_y} - {label_x}\n"
                     f"n={len(sub)}, CCC={ccc:.3f}, bias={bias:.2f} mL", fontsize=8.5)

        agreement_rows.append({'comparison': f"{label_y} vs {label_x}", 'n': len(sub), 'CCC': ccc,
                               'pearson_r': pearson_r, 'bias_mL': bias, 'sd_diff_mL': sd_diff,
                               'LoA_lower_mL': loa_lo, 'LoA_upper_mL': loa_hi})

    # NOT: baslik gomulmuyor (dergi kurali) -- panel basliklari karsilastirilan
    # cifti zaten (kisa etiketlerle) belirtiyor.
    plt.tight_layout()
    path = save_figure(fig, "FigureS1_bland_altman_volume_agreement")
    save_table(pd.DataFrame(agreement_rows), "FigureS1_agreement_metrics")
    return path


# ============================================================
# FIGURE S2 - CALIBRATION + DECISION-CURVE ANALYSIS
# ============================================================
def build_figure_s2(cohort_b, table_s2_context):
    print_header("FIGURE S2 - Calibration and decision-curve analysis")
    pred_a, pred_ap, valid = table_s2_context['pred_a'], table_s2_context['pred_ap'], table_s2_context['valid']
    y = cohort_b['csPCa_binary'].values[valid]
    pa, pap = pred_a[valid], pred_ap[valid]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Panel A: calibration (decile bins, OOF predictions)
    for pred, label, marker in [(pa, 'Model A', 'o'), (pap, "Model A'", 's')]:
        bins = pd.qcut(pred, q=10, duplicates='drop')
        grp = pd.DataFrame({'pred': pred, 'y': y, 'bin': bins}).groupby('bin', observed=True)
        mean_pred = grp['pred'].mean().values
        obs_frac = grp['y'].mean().values
        axes[0].plot(mean_pred, obs_frac, marker=marker, linestyle='-', linewidth=1.3,
                     markersize=5, label=label)
    lims = [0, max(pa.max(), pap.max()) * 1.05]
    axes[0].plot(lims, lims, linestyle='--', color='gray', linewidth=1, label='Ideal')
    axes[0].set_xlabel('Mean predicted probability (decile bin)')
    axes[0].set_ylabel('Observed csPCa fraction')
    axes[0].set_title(f"A. Calibration (repeated-CV out-of-fold, n={len(y)})", fontsize=9.5)
    axes[0].legend(loc='upper left', fontsize=8)

    # Panel B: DCA curve
    nb_a = ph5.net_benefit_curve(y, pa, ph5.DCA_THRESHOLD_RANGE)
    nb_ap = ph5.net_benefit_curve(y, pap, ph5.DCA_THRESHOLD_RANGE)
    nb_all = ph5.treat_all_curve(y.mean(), ph5.DCA_THRESHOLD_RANGE)
    axes[1].plot(ph5.DCA_THRESHOLD_RANGE, nb_a, linewidth=1.6, label='Model A')
    axes[1].plot(ph5.DCA_THRESHOLD_RANGE, nb_ap, linewidth=1.6, label="Model A'")
    axes[1].plot(ph5.DCA_THRESHOLD_RANGE, nb_all, linewidth=1.2, linestyle='--', color='gray', label='Treat all')
    axes[1].axhline(0, color='black', linewidth=1, linestyle=':', label='Treat none')
    axes[1].set_xlabel('Threshold probability')
    axes[1].set_ylabel('Net benefit')
    axes[1].set_title(f"B. Decision-curve analysis (n={len(y)})", fontsize=9.5)
    axes[1].legend(loc='upper right', fontsize=8)

    # NOT: baslik gomulmuyor (dergi kurali) -- panel A/B alt-basliklari kaliyor.
    plt.tight_layout()
    path = save_figure(fig, "FigureS2_calibration_and_DCA")

    dca_curve_df = pd.DataFrame({'threshold': ph5.DCA_THRESHOLD_RANGE, 'nb_model_a': nb_a,
                                 'nb_model_ap': nb_ap, 'nb_treat_all': nb_all})
    save_table(dca_curve_df, "FigureS2_dca_full_curve")
    return path


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print_header("PHASE 7: TABLO VE FIGUR URETIMI BASLADI")
    df, cohort_a, cohort_b = load_data()

    table1_df = build_table1(cohort_a, cohort_b)
    table2_raw, table2_display, table2_context = build_table2(df, cohort_a, cohort_b)
    table3_coef, table3_ratio, table3_context = build_table3(cohort_b, table2_context)

    build_figure1(df, cohort_a, cohort_b)
    build_figure2(cohort_a, cohort_b, table2_context)

    build_table_s1(df, cohort_a, cohort_b)
    table_s2_context = build_table_s2(cohort_a, cohort_b)
    build_table_s3(df, cohort_a, cohort_b, table2_context)

    build_figure_s1(df)
    build_figure_s2(cohort_b, table_s2_context)

    print_header("PHASE 7 TAMAMLANDI")
    print(f"Tablolar: {TABLES_DIR}")
    print(f"Figurler: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
