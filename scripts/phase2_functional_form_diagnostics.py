"""
Revizyon Phase 2 - Fonksiyonel Form Kontrolu (Spline, Interaction, Collinearity/VIF)
=================================================================================
Amac: Model A / Model A' icin varsayilan log-lineer form (log(PSA), log(hacim))
veriye uygun mu? Bu, DIAGNOSTIC bir faz -- primer performans sonucu URETMIYOR,
sadece "katsayi orani ve PSAD-esitlik LRT'si ne kadar sert yorumlanabilir"
sorusunun cevabini hazirliyor. Phase 3'ten (primer analizler) ONCE calisiyor,
bloker.

Kohort: Phase 1'de kilitlenen Cohort B (N=1096, RUMC/ZGT, reported PSAD
zorunlu degil, PCNN haric). Bu script Cohort B'yi Phase 1 ile BIREBIR AYNI
kriterle kendi icinde yeniden kuruyor (self-contained convention, projedeki
diger fazlarla ayni pratik).

KRITIK TASARIM KARARI -- neden patsy DEGIL manuel Harrell RCS basis:
patsy'nin cr()/bs() fonksiyonlari orthogonalized bir spline basis'i
kullanir; bu basis'te nonlinear terimlerin katsayilarini sifira ayarlamak
DOGRUSAL modele TAM OLARAK esit degildir (farkli parametrizasyon). Bu
yuzden patsy-tabanli bir LRT "nested" degildir, formal olarak gecersiz
sayilabilir. Harrell'in "truncated power basis" formulasyonu (Regression
Modeling Strategies) ise linear terimi spline basis'inin ICINDE birebir
icerir -- nonlinear terimler sifirlanirsa TAM OLARAK linear modele
indirgenir. Bu yuzden LRT burada GERCEKTEN nested ve gecerli.

Uc teknik uyari uygulandi:
1. Knot sayisi N'e gore adaptif: 3 knot ile basla, delta_AIC>4 ise 4 knot da
   dene; 3 ve 4 knot arasi fark <2 ise parsimony geregi 3 knotta kal.
2. Her degisken izole test edilir (logPSA spline'lanirken hacim LINEAR
   kalir, ve tersi) -- hangi degiskenin sorunlu oldugu boylece ayirt edilir.
   Ayrica (promptun ayrica istedigi) "her ikisi de spline" kombine testi de
   yapiliyor, ama bu sabit 3-knotla, sadece bilgi amacli/ikincil.
3. VIF, INTERCEPT DAHIL tasarim matrisi uzerinde hesaplaniyor (statsmodels'in
   variance_inflation_factor fonksiyonunun dogru calismasi icin sart --
   intercept olmadan VIF yapay olarak sisebilir), ama raporlanan tabloda
   intercept satiri GOSTERILMIYOR.
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import chi2, pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

DATASET_ROOT = "/content/drive/MyDrive/Uroloji_Projesi/PICAI_Dataset"
MERGED_CSV_PATH = os.path.join(DATASET_ROOT, "QA_outputs", "faz2_merged_clinical_ai_volume.csv")
OUTPUT_DIR = os.path.join(DATASET_ROOT, "QA_outputs", "revision_outputs")

DELTA_AIC_TRY_4KNOTS_THRESHOLD = 4.0
DELTA_AIC_PARSIMONY_THRESHOLD = 2.0
EXPECTED_N = 1096


def print_header(title):
    print("\n" + "=" * 90)
    print(f"||  {title}")
    print("=" * 90)


# ============================================================
# COHORT B (Phase 1 ile birebir ayni kriter -- self-contained)
# ============================================================
def build_cohort_b():
    df = pd.read_csv(MERGED_CSV_PATH)
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
    # DUZELTME: log1p (=log(1+x)) DEGIL, gercek log(x) kullanildi. Gerekce: conventional
    # PSAD constraint'i log(PSA/hacim) = log(PSA) - log(hacim) matematiksel olarak SADECE
    # gercek log ile tam esdegerdir; log1p ile log(1+PSA) - log(1+hacim) != log((1+PSA)/(1+hacim)),
    # yani "esit-agirlik kisiti" testinin matematiksel yorumu bozulur. log1p'nin sifir/negatif
    # deger korumasi zaten gereksiz -- asagida kontrol ediliyor (PSA/hacim burada hep pozitif).
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
    n_events = int(cohort['csPCa_binary'].sum())
    print(f"Cohort B (RUMC/ZGT, reported PSAD zorunlu degil): N={n}, events={n_events} "
          f"({100 * n_events / n:.1f}%)")
    if n != EXPECTED_N:
        print(f"UYARI: beklenen N={EXPECTED_N} degil, N={n} -- Phase 1'deki CSV ile ayni mi kontrol et.")
    return cohort


# ============================================================
# LOJISTIK REGRESYON YARDIMCILARI
# ============================================================
def fit_logit(y, X):
    model = sm.Logit(y, X).fit(disp=0)
    if not model.mle_retvals.get('converged', True):
        raise RuntimeError("Model yakinsamadi.")
    return model


def rcs_basis(x, knots):
    """
    Harrell restricted cubic spline (truncated power basis). k knot icin
    k-1 kolon doner: [x_linear, nonlinear_1, ..., nonlinear_{k-2}].
    """
    x = np.asarray(x, dtype=float)
    knots = np.asarray(sorted(knots), dtype=float)
    k = len(knots)
    if k < 3:
        raise ValueError("En az 3 knot gerekli.")
    tk, tk1 = knots[-1], knots[-2]
    cols = [x]
    for j in range(k - 2):
        tj = knots[j]
        term = (
            np.clip(x - tj, 0, None) ** 3
            - np.clip(x - tk1, 0, None) ** 3 * (tk - tj) / (tk - tk1)
            + np.clip(x - tk, 0, None) ** 3 * (tk1 - tj) / (tk - tk1)
        )
        cols.append(term)
    return np.column_stack(cols)


def get_knots(x, n_knots):
    percentiles = {3: [5, 50, 95], 4: [5, 35, 65, 95]}[n_knots]
    return np.percentile(x, percentiles)


# ============================================================
# 1. SPLINE / FONKSIYONEL FORM TESTLERI
# ============================================================
def spline_test_single_variable(y, age, target_log, other_log, centre, target_name, n_knots=3):
    """target_log SPLINE, other_log LINEAR (izole test -- hangi degiskenin
    sorunlu oldugunu ayirt etmek icin)."""
    knots = get_knots(target_log, n_knots)

    X_lin = sm.add_constant(np.column_stack([age, target_log, other_log, centre]))
    model_lin = fit_logit(y, X_lin)

    spline_cols = rcs_basis(target_log, knots)
    X_spline = sm.add_constant(np.column_stack([age, spline_cols, other_log, centre]))
    model_spline = fit_logit(y, X_spline)

    df_diff = X_spline.shape[1] - X_lin.shape[1]
    lr_chi2 = 2 * (model_spline.llf - model_lin.llf)
    p_value = chi2.sf(lr_chi2, df=df_diff)

    pred_lin = model_lin.predict(X_lin)
    pred_spline = model_spline.predict(X_spline)
    auc_lin = roc_auc_score(y, pred_lin)
    auc_spline = roc_auc_score(y, pred_spline)

    return {
        'tested_variable': target_name, 'knots': n_knots,
        'knot_locations': list(np.round(knots, 4)),
        'n': len(y), 'events': int(np.sum(y)),
        'll_linear': model_lin.llf, 'll_spline': model_spline.llf,
        'lr_chi2': lr_chi2, 'df': df_diff, 'p_value': p_value,
        'aic_linear': model_lin.aic, 'aic_spline': model_spline.aic,
        # aic_improvement_with_spline = AIC_linear - AIC_spline: POZITIF = spline daha iyi (AIC dusuk),
        # isim direkt yonu belirtiyor, "delta_aic" gibi belirsiz degil.
        'aic_improvement_with_spline': model_lin.aic - model_spline.aic,
        'auc_linear': auc_lin, 'auc_spline': auc_spline,
        'delta_auc': auc_spline - auc_lin,
        'spline_coefs': np.asarray(model_spline.params)[1:1 + spline_cols.shape[1]],
    }


def spline_test_both_variables(y, age, psa_log, vol_log, centre, vol_name, n_knots=3):
    knots_psa = get_knots(psa_log, n_knots)
    knots_vol = get_knots(vol_log, n_knots)

    X_lin = sm.add_constant(np.column_stack([age, psa_log, vol_log, centre]))
    model_lin = fit_logit(y, X_lin)

    spline_psa = rcs_basis(psa_log, knots_psa)
    spline_vol = rcs_basis(vol_log, knots_vol)
    X_spline = sm.add_constant(np.column_stack([age, spline_psa, spline_vol, centre]))
    model_spline = fit_logit(y, X_spline)

    df_diff = X_spline.shape[1] - X_lin.shape[1]
    lr_chi2 = 2 * (model_spline.llf - model_lin.llf)
    p_value = chi2.sf(lr_chi2, df=df_diff)

    pred_lin = model_lin.predict(X_lin)
    pred_spline = model_spline.predict(X_spline)
    auc_lin = roc_auc_score(y, pred_lin)
    auc_spline = roc_auc_score(y, pred_spline)

    return {
        'tested_variable': f'both (log_PSA + {vol_name})', 'knots': n_knots,
        'knot_locations': f"PSA={list(np.round(knots_psa, 3))}, vol={list(np.round(knots_vol, 3))}",
        'n': len(y), 'events': int(np.sum(y)),
        'll_linear': model_lin.llf, 'll_spline': model_spline.llf,
        'lr_chi2': lr_chi2, 'df': df_diff, 'p_value': p_value,
        'aic_linear': model_lin.aic, 'aic_spline': model_spline.aic,
        'aic_improvement_with_spline': model_lin.aic - model_spline.aic,
        'auc_linear': auc_lin, 'auc_spline': auc_spline,
        'delta_auc': auc_spline - auc_lin,
    }


def adaptive_spline_test(y, age, target_log, other_log, centre, target_name):
    """3 knot dene; ciddi iyilesme varsa (aic_improvement_with_spline>4) 4 knot da dene,
    parsimony icin fark <2 ise 3 knotta kal."""
    print(f"\n  -- {target_name}: 3-knot spline testi --")
    result_3 = spline_test_single_variable(y, age, target_log, other_log, centre, target_name, n_knots=3)
    print(f"     AIC_improvement(3-knot) = {result_3['aic_improvement_with_spline']:.3f} "
          f"(pozitif=spline daha iyi), p={result_3['p_value']:.4g}, "
          f"knot yerleri={result_3['knot_locations']}")

    result_3['knot_selection_note'] = "3 knot (baslangic)"

    if result_3['aic_improvement_with_spline'] > DELTA_AIC_TRY_4KNOTS_THRESHOLD:
        print(f"     AIC_improvement>{DELTA_AIC_TRY_4KNOTS_THRESHOLD} -- 4-knot da deneniyor...")
        result_4 = spline_test_single_variable(y, age, target_log, other_log, centre, target_name, n_knots=4)
        print(f"     AIC_improvement(4-knot) = {result_4['aic_improvement_with_spline']:.3f}, "
              f"p={result_4['p_value']:.4g}")
        diff_3_4 = abs(result_3['aic_spline'] - result_4['aic_spline'])
        if diff_3_4 < DELTA_AIC_PARSIMONY_THRESHOLD:
            print(f"     |AIC_3knot - AIC_4knot|={diff_3_4:.3f} < {DELTA_AIC_PARSIMONY_THRESHOLD} "
                  f"-- parsimony: 3 knotta kaliniyor.")
            result_3['knot_selection_note'] = f"3 knot secildi (parsimony; 4-knot denendi, fark={diff_3_4:.3f}<2)"
            result_3['four_knot_aic_improvement'] = result_4['aic_improvement_with_spline']
            return result_3
        else:
            print(f"     |AIC_3knot - AIC_4knot|={diff_3_4:.3f} >= {DELTA_AIC_PARSIMONY_THRESHOLD} "
                  f"-- 4-knot tercih ediliyor.")
            result_4['knot_selection_note'] = f"4 knot secildi (3-knota gore belirgin iyilesme, fark={diff_3_4:.3f})"
            return result_4
    else:
        result_3['knot_selection_note'] = "3 knot yeterli (esik asilmadi, 4-knot denenmedi)"
        return result_3


# ============================================================
# 2. INTERACTION TESTLERI
# ============================================================
def interaction_test(y, age, psa_log, vol_log, centre, vol_name):
    X_base = sm.add_constant(np.column_stack([age, psa_log, vol_log, centre]))
    model_base = fit_logit(y, X_base)

    interaction_term = psa_log * vol_log
    X_int = sm.add_constant(np.column_stack([age, psa_log, vol_log, interaction_term, centre]))
    model_int = fit_logit(y, X_int)

    idx_int = 4  # const(0), age(1), psa(2), vol(3), interaction(4), centre(5)
    coef = np.asarray(model_int.params)[idx_int]
    se = np.asarray(model_int.bse)[idx_int]
    p = np.asarray(model_int.pvalues)[idx_int]
    or_val = np.exp(coef)
    ci_lo, ci_hi = np.exp(coef - 1.96 * se), np.exp(coef + 1.96 * se)

    lr_chi2 = 2 * (model_int.llf - model_base.llf)
    p_lrt = chi2.sf(lr_chi2, df=1)

    pred_base = model_base.predict(X_base)
    pred_int = model_int.predict(X_int)
    auc_base = roc_auc_score(y, pred_base)
    auc_int = roc_auc_score(y, pred_int)

    return {
        'model': vol_name, 'interaction_coef': coef, 'interaction_OR': or_val,
        'ci_low': ci_lo, 'ci_high': ci_hi, 'wald_p': p,
        'lr_chi2': lr_chi2, 'df': 1, 'lrt_p': p_lrt,
        'aic_base': model_base.aic, 'aic_interaction': model_int.aic,
        # aic_improvement_with_interaction = AIC_base - AIC_interaction: POZITIF = interaction daha iyi.
        'aic_improvement_with_interaction': model_base.aic - model_int.aic,
        'auc_base': auc_base, 'auc_interaction': auc_int,
        'delta_auc': auc_int - auc_base,
    }


# ============================================================
# 3. COLLINEARITY / VIF
# ============================================================
def collinearity_and_vif(age, psa_log, vol_log, centre, model_name, vol_label):
    r, p_r = pearsonr(psa_log, vol_log)
    rho, p_rho = spearmanr(psa_log, vol_log)
    print(f"\n  {model_name}: Pearson(logPSA,{vol_label})={r:.4f} (p={p_r:.3g}), "
          f"Spearman={rho:.4f} (p={p_rho:.3g})")

    # VIF intercept DAHIL matrix uzerinde hesaplaniyor (statsmodels'in dogru
    # calismasi icin sart -- intercept olmadan VIF yapay sisebilir), ama
    # raporlanan tabloda intercept satiri GOSTERILMIYOR.
    X = sm.add_constant(np.column_stack([age, psa_log, vol_log, centre]))
    var_names = ['const', 'age_per_10', 'log_PSA', vol_label, 'centre_RUMC']
    vif_data = []
    for i, name in enumerate(var_names):
        if name == 'const':
            continue
        vif = variance_inflation_factor(X, i)
        vif_data.append({'model': model_name, 'variable': name, 'VIF': vif})
        print(f"    VIF({name}) = {vif:.3f}")

    corr_row = {'model': model_name, 'pearson_r': r, 'pearson_p': p_r,
                'spearman_rho': rho, 'spearman_p': p_rho}
    return corr_row, vif_data


# ============================================================
# 4. PARTIAL-EFFECT (SPLINE) GRAFIKLERI
# ============================================================
def plot_partial_effect(x_raw, spline_coefs, knots, xlabel, out_path):
    x_grid = np.linspace(np.percentile(x_raw, 1), np.percentile(x_raw, 99), 200)
    basis_grid = rcs_basis(x_grid, knots)
    partial_log_odds = basis_grid @ spline_coefs

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(x_grid, partial_log_odds, color='tab:blue', linewidth=2)
    rug_y = np.full_like(x_raw, partial_log_odds.min() - 0.3, dtype=float)
    ax.plot(x_raw, rug_y, '|', color='gray', alpha=0.3, markersize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Partial log-odds (spline fit)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Grafik kaydedildi: {out_path}")


# ============================================================
# 5. YORUM / KARAR
# ============================================================
def decide_interpretation(spline_rows, interaction_rows, max_vif):
    max_delta_auc_spline = max(abs(r['delta_auc']) for r in spline_rows)
    max_delta_aic_spline = max(r['aic_improvement_with_spline'] for r in spline_rows)
    max_delta_auc_int = max(abs(r['delta_auc']) for r in interaction_rows)
    max_delta_aic_int = max(r['aic_improvement_with_interaction'] for r in interaction_rows)

    material_auc_change = max(max_delta_auc_spline, max_delta_auc_int) > 0.01
    strong_aic_change = max(max_delta_aic_spline, max_delta_aic_int) > 10
    minor_aic_change = max(max_delta_aic_spline, max_delta_aic_int) > 4
    vif_concern = max_vif > 5

    if material_auc_change or vif_concern:
        return ("Strong nonlinearity or interaction; coefficient ratio should not be emphasized",
                f"max|deltaAUC|={max(max_delta_auc_spline, max_delta_auc_int):.4f}, max_VIF={max_vif:.2f}")
    elif strong_aic_change:
        return ("Nonlinearity present; coefficient ratio should be interpreted cautiously",
                f"max_deltaAIC={max(max_delta_aic_spline, max_delta_aic_int):.2f}")
    elif minor_aic_change:
        return ("Minor nonlinearity, no material performance impact",
                f"max_deltaAIC={max(max_delta_aic_spline, max_delta_aic_int):.2f}, "
                f"max|deltaAUC|={max(max_delta_auc_spline, max_delta_auc_int):.4f}")
    else:
        return ("Log-linear form acceptable",
                f"max_deltaAIC={max(max_delta_aic_spline, max_delta_aic_int):.2f}, max_VIF={max_vif:.2f}")


SENTENCES = {
    "Log-linear form acceptable":
        "Restricted cubic spline terms for log-transformed PSA and prostate volume did not "
        "materially improve model fit compared with the prespecified log-linear specification, "
        "and no clinically meaningful PSA-volume interaction was observed. Correlation and VIF "
        "diagnostics did not indicate problematic collinearity. Therefore, coefficient-ratio and "
        "restricted-model analyses were interpreted within the prespecified log-linear framework.",
    "Minor nonlinearity, no material performance impact":
        "Restricted cubic spline terms showed statistically detectable but practically minor "
        "departures from log-linearity; AIC and discrimination changes were small. Coefficient-ratio "
        "estimates were retained as the primary interpretation, with this minor nonlinearity noted "
        "as a limitation.",
    "Nonlinearity present; coefficient ratio should be interpreted cautiously":
        "Functional-form diagnostics showed some evidence of nonlinearity; however, flexible spline "
        "terms produced minimal changes in discrimination. Coefficient-ratio estimates were therefore "
        "retained as descriptive, model-dependent summaries rather than interpreted as transportable "
        "biological weights.",
    "Strong nonlinearity or interaction; coefficient ratio should not be emphasized":
        "Because flexible terms materially improved model fit and/or interaction terms changed the "
        "PSA-volume relationship, coefficient-ratio estimates from the simple log-linear model should "
        "not be emphasized as primary evidence. The restricted-vs-free model comparison may still be "
        "reported, but its interpretation must be explicitly model-form-dependent.",
}


def write_report(cohort, spline_df, interaction_df, corr_df, vif_df, category, reason, out_dir):
    n, events = len(cohort), int(cohort['csPCa_binary'].sum())
    lines = []
    lines.append("# Phase 2 - Functional Form Report\n")
    lines.append(f"**Cohort**: Cohort B (RUMC/ZGT), N={n}, events={events}\n")
    lines.append("## 1. Spline (functional form) checks\n")
    lines.append("```\n" + spline_df.drop(columns=['spline_coefs'], errors='ignore').to_string(index=False) + "\n```")
    lines.append("\n## 2. Interaction tests\n")
    lines.append("```\n" + interaction_df.to_string(index=False) + "\n```")
    lines.append("\n## 3. Collinearity\n")
    lines.append("```\n" + corr_df.to_string(index=False) + "\n```")
    lines.append("\n## VIF\n")
    lines.append("```\n" + vif_df.to_string(index=False) + "\n```")
    lines.append(f"\n## 4. Final interpretation category\n\n**{category}**\n\n(Gerekce: {reason})\n")
    lines.append("\n## 5. Suggested manuscript sentence\n\n" + SENTENCES[category] + "\n")

    report_path = os.path.join(out_dir, "phase2_functional_form_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nRapor kaydedildi: {report_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print_header("PHASE 2: FONKSIYONEL FORM KONTROLU (Cohort B, N=1096, RUMC/ZGT)")
    cohort = build_cohort_b()

    y = cohort['csPCa_binary'].values
    age = cohort['age_per_10'].values
    psa = cohort['log_PSA'].values
    vol_manual = cohort['log_manual_volume'].values
    vol_ai = cohort['log_AI_volume'].values
    centre = cohort['centre_RUMC'].values

    spline_rows = []

    print_header("1. SPLINE TESTLERI — MODEL A (manuel hacim)")
    r1 = adaptive_spline_test(y, age, psa, vol_manual, centre, 'log_PSA (Model A)')
    r1['model_family'] = 'Model A'
    spline_rows.append(r1)
    r2 = adaptive_spline_test(y, age, vol_manual, psa, centre, 'log_manual_volume (Model A)')
    r2['model_family'] = 'Model A'
    spline_rows.append(r2)
    r3 = spline_test_both_variables(y, age, psa, vol_manual, centre, 'log_manual_volume', n_knots=3)
    r3['model_family'] = 'Model A'
    r3['knot_selection_note'] = '3 knot (sabit, kombine/ikincil test)'
    spline_rows.append(r3)

    print_header("1. SPLINE TESTLERI — MODEL A' (AI hacim)")
    r4 = adaptive_spline_test(y, age, psa, vol_ai, centre, "log_PSA (Model A')")
    r4['model_family'] = "Model A'"
    spline_rows.append(r4)
    r5 = adaptive_spline_test(y, age, vol_ai, psa, centre, "log_AI_volume (Model A')")
    r5['model_family'] = "Model A'"
    spline_rows.append(r5)
    r6 = spline_test_both_variables(y, age, psa, vol_ai, centre, 'log_AI_volume', n_knots=3)
    r6['model_family'] = "Model A'"
    r6['knot_selection_note'] = '3 knot (sabit, kombine/ikincil test)'
    spline_rows.append(r6)

    spline_df = pd.DataFrame(spline_rows)
    print("\n" + spline_df[['model_family', 'tested_variable', 'knots', 'n', 'events', 'lr_chi2', 'df',
                             'p_value', 'aic_linear', 'aic_spline', 'aic_improvement_with_spline',
                             'auc_linear', 'auc_spline', 'delta_auc',
                             'knot_selection_note']].to_string(index=False))
    spline_df.drop(columns=['spline_coefs']).to_csv(
        os.path.join(OUTPUT_DIR, "phase2_spline_functional_form_checks.csv"), index=False)

    print_header("2. INTERACTION TESTLERI")
    int_manual = interaction_test(y, age, psa, vol_manual, centre, 'Model A (manual volume)')
    int_ai = interaction_test(y, age, psa, vol_ai, centre, "Model A' (AI volume)")
    interaction_df = pd.DataFrame([int_manual, int_ai])
    print(interaction_df.to_string(index=False))
    interaction_df.to_csv(os.path.join(OUTPUT_DIR, "phase2_interaction_tests.csv"), index=False)

    print_header("3. COLLINEARITY / VIF")
    corr_a, vif_a = collinearity_and_vif(age, psa, vol_manual, centre, 'Model A', 'log_manual_volume')
    corr_ai, vif_ai = collinearity_and_vif(age, psa, vol_ai, centre, "Model A'", 'log_AI_volume')
    corr_df = pd.DataFrame([corr_a, corr_ai])
    vif_df = pd.DataFrame(vif_a + vif_ai)
    corr_df.to_csv(os.path.join(OUTPUT_DIR, "phase2_collinearity_correlations.csv"), index=False)
    vif_df.to_csv(os.path.join(OUTPUT_DIR, "phase2_vif_diagnostics.csv"), index=False)
    max_vif = vif_df['VIF'].max()

    print_header("4. PARTIAL-EFFECT GRAFIKLERI (Model A' spline sonuclarindan)")
    try:
        knots_psa = get_knots(psa, r4['knots'])
        plot_partial_effect(psa, r4['spline_coefs'], knots_psa, 'log(PSA)',
                             os.path.join(OUTPUT_DIR, "supp_functional_form_log_psa.png"))
        knots_manual = get_knots(vol_manual, r2['knots'])
        plot_partial_effect(vol_manual, r2['spline_coefs'], knots_manual, 'log(manual volume)',
                             os.path.join(OUTPUT_DIR, "supp_functional_form_log_manual_volume.png"))
        knots_ai = get_knots(vol_ai, r5['knots'])
        plot_partial_effect(vol_ai, r5['spline_coefs'], knots_ai, 'log(AI volume)',
                             os.path.join(OUTPUT_DIR, "supp_functional_form_log_ai_volume.png"))
    except Exception as e:
        print(f"UYARI: grafik uretimi basarisiz oldu ({e}), CSV ciktilari yine de gecerli.")

    print_header("5. YORUM / KARAR")
    category, reason = decide_interpretation(spline_rows, [int_manual, int_ai], max_vif)
    print(f"SONUC KATEGORISI: {category}")
    print(f"Gerekce: {reason}")

    write_report(cohort, spline_df, interaction_df, corr_df, vif_df, category, reason, OUTPUT_DIR)

    print_header("PHASE 2 TAMAMLANDI")
    print("Not: Phase 3'e (coefficient-ratio bootstrap, LRT) gecmeden bu sonuclar")
    print("degerlendirilecek -- yukaridaki kategori, katsayi oraninin ne kadar sert")
    print("yorumlanabilecegini belirleyecek.")


if __name__ == "__main__":
    main()
