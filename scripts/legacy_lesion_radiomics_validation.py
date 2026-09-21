"""
PI-CAI Faz 6b - Track B Ic Dogrulama (Lezyon Radyomigi ile GG3+ Tahmini)
==============================================================================
Faz 3 Track B'nin in-sample sonucunu (AUC=0.613, N~411) Faz 6'daki ayni
yontemlerle (Harrell bootstrap optimism-correction + tekrarli stratified
k-fold) resmi olarak doguluyor. Faz 6'dan farkli olarak burada IKI modelin
KARSILASTIRMASI degil, TEK bir modelin (yas + lezyon hacmi + min-ADC +
ADC-orani -> GG3+) kendi optimism-corrected performansi degerlendiriliyor.

Girdi: Faz 3'un Track B sonunda kaydettigi
`{DATASET_ROOT}/QA_outputs/faz3_trackb_lesion_features.csv`. Goruntuye
ihtiyac yok, saf istatistik islemidir.

N~411 ve 4 parametre (hasta/parametre orani ~103:1), Faz 2/6'daki Model
A/A' N~1031/4-param (~258:1) modeline gore daha dusuk -- bu yuzden burada
Faz 6'dakinden daha buyuk bir optimism (asiri uyum) beklenebilir.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
import statsmodels.api as sm
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
TRACK_B_CSV_PATH = "/content/drive/MyDrive/Uroloji_Projesi/PICAI_Dataset/QA_outputs/faz3_trackb_lesion_features.csv"

FEATURES_RADIOMICS = ['age_10yr', 'lesion_vol_log', 'lesion_adc_min', 'lesion_adc_ratio']

N_BOOTSTRAP = 500
N_KFOLD_SPLITS = 5
N_KFOLD_REPEATS = 20
RANDOM_SEED = 42
SEPARATION_AUC_GUARD = 0.999


# ============================================================
# UTILITIES
# ============================================================
def print_header(title):
    print("\n" + "=" * 80)
    print(f"||  {title}")
    print("=" * 80)


def print_subheader(title):
    print(f"\n{'-' * 70}")
    print(f"  {title}")
    print(f"{'-' * 70}")


def prepare_cohort(df):
    """Faz 3 Track B'nin 'model_df' tanimini birebir yeniden olusturur."""
    df = df.copy()
    model_df = df.dropna(subset=['lesion_volume_mL', 'lesion_adc_min', 'lesion_adc_ratio', 'patient_age']).copy()
    model_df['age_10yr'] = model_df['patient_age'] / 10.0
    model_df['lesion_vol_log'] = np.log1p(model_df['lesion_volume_mL'])
    model_df['high_grade'] = (model_df['case_ISUP'] >= 3).astype(int)
    return model_df.reset_index(drop=True)


def fit_logit(train_df, feature_cols, y_col='high_grade'):
    X = train_df[feature_cols].copy()
    X.insert(0, 'const', 1.0)
    y = train_df[y_col].astype(int)
    model = sm.Logit(y, X).fit(disp=0)
    if not model.mle_retvals.get('converged', True):
        # statsmodels 'perfect separation'da ya da baska nedenlerle yakinsamayinca
        # exception firlatmaz -- bunu burada RuntimeError'a cevirip mevcut
        # try/except bloklarinin (bootstrap/k-fold donguleri) yakalamasini sagliyoruz.
        raise RuntimeError("Logit fit yakinsamadi (mle_retvals['converged']=False)")
    return model


def predict_logit(model, eval_df, feature_cols):
    X = eval_df[feature_cols].copy()
    X.insert(0, 'const', 1.0)
    return model.predict(X)


def is_degenerate_auc(*auc_values):
    """statsmodels 'perfect separation' durumunda exception FIRLATMAZ, sadece
    (bastirilmis) uyari basip anlamsiz katsayilarla 'basarili' doner."""
    return any((a >= SEPARATION_AUC_GUARD or a <= 1 - SEPARATION_AUC_GUARD) for a in auc_values)


# ============================================================
# SECTION 1: BOOTSTRAP OPTIMISM-CORRECTION (tek model)
# ============================================================
def bootstrap_optimism_single(df, feature_cols, y_col='high_grade'):
    n_bootstrap = N_BOOTSTRAP
    seed = RANDOM_SEED
    rng = np.random.RandomState(seed)
    n = len(df)
    y = df[y_col].astype(int).values

    try:
        model_full = fit_logit(df, feature_cols, y_col)
        auc_apparent = roc_auc_score(y, predict_logit(model_full, df, feature_cols))
    except Exception as e:
        print(f"\n   UYARI: Tam veri (apparent) model fit basarisiz: {e}")
        return None
    if is_degenerate_auc(auc_apparent):
        print(f"\n   UYARI: Tam veri modelinde AUC~{auc_apparent:.3f} -- olasi perfect separation. Atlaniyor.")
        return None

    optimism_list = []
    n_failed = 0

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        boot_df = df.iloc[idx].reset_index(drop=True)
        if boot_df[y_col].nunique() < 2:
            n_failed += 1
            continue
        try:
            model_boot = fit_logit(boot_df, feature_cols, y_col)
            pred_boot = predict_logit(model_boot, boot_df, feature_cols)
            pred_orig = predict_logit(model_boot, df, feature_cols)
        except Exception:
            n_failed += 1
            continue

        auc_boot = roc_auc_score(boot_df[y_col].astype(int), pred_boot)
        auc_orig = roc_auc_score(y, pred_orig)

        if is_degenerate_auc(auc_boot, auc_orig):
            n_failed += 1
            continue

        optimism_list.append(auc_boot - auc_orig)

    print(f"\n   Bootstrap tekrar sayisi: {n_bootstrap} | basarisiz/atlanan: {n_failed}")
    if len(optimism_list) == 0:
        print("   UYARI: Tum bootstrap iterasyonlari basarisiz oldu -- sonuc hesaplanamiyor.")
        return None

    mean_optimism = np.mean(optimism_list)
    corrected_samples = auc_apparent - np.array(optimism_list)
    ci_low, ci_high = np.percentile(corrected_samples, [2.5, 97.5])

    return {
        'n_used': len(optimism_list), 'n_failed': n_failed,
        'auc_apparent': auc_apparent, 'optimism': mean_optimism,
        'auc_corrected': auc_apparent - mean_optimism,
        'ci_low': ci_low, 'ci_high': ci_high,
    }


# ============================================================
# SECTION 2: TEKRARLI STRATIFIED K-FOLD (tek model)
# ============================================================
def repeated_kfold_single(df, feature_cols, y_col='high_grade'):
    n_splits, n_repeats, seed = N_KFOLD_SPLITS, N_KFOLD_REPEATS, RANDOM_SEED
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    y = df[y_col].astype(int).values

    fold_aucs = []
    n_failed = 0

    for train_idx, test_idx in rkf.split(df, y):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        if train_df[y_col].nunique() < 2 or test_df[y_col].nunique() < 2:
            n_failed += 1
            continue
        try:
            model = fit_logit(train_df, feature_cols, y_col)
            pred = predict_logit(model, test_df, feature_cols)
        except Exception:
            n_failed += 1
            continue
        auc_fold = roc_auc_score(test_df[y_col], pred)
        if is_degenerate_auc(auc_fold):
            n_failed += 1
            continue
        fold_aucs.append(auc_fold)

    total_folds = n_splits * n_repeats
    print(f"\n   Toplam fold sayisi: {total_folds} | basarisiz/atlanan: {n_failed}")
    if len(fold_aucs) == 0:
        print("   UYARI: Tum fold'lar basarisiz oldu -- sonuc hesaplanamiyor.")
        return None

    return {
        'n_used': len(fold_aucs), 'n_failed': n_failed,
        'auc_mean': np.mean(fold_aucs), 'auc_sd': np.std(fold_aucs),
        'ci_low': np.percentile(fold_aucs, 2.5), 'ci_high': np.percentile(fold_aucs, 97.5),
    }


# ============================================================
# MAIN
# ============================================================
def run_faz6b():
    print("\n" + "#" * 80)
    print("#" + " " * 10 + "PI-CAI FAZ 6b - TRACK B IC DOGRULAMA (GG3+ RADYOMIK)" + " " * 10 + "#")
    print("#" * 80)

    print_header("STEP 0: VERI YUKLEME")
    df = pd.read_csv(TRACK_B_CSV_PATH)
    model_df = prepare_cohort(df)
    n_pos = model_df['high_grade'].sum()
    print(f"\n   Kohort N = {len(model_df)}, GG3+ = {n_pos} ({100*n_pos/len(model_df):.1f}%)")
    print(f"   Model: high_grade ~ " + " + ".join(FEATURES_RADIOMICS))

    print_header("SECTION 1: BOOTSTRAP OPTIMISM-CORRECTION")
    boot_result = bootstrap_optimism_single(model_df, FEATURES_RADIOMICS)
    if boot_result:
        print(f"\n   Apparent (in-sample) AUC : {boot_result['auc_apparent']:.3f}")
        print(f"   Optimism (asiri uyum)    : {boot_result['optimism']:.4f}")
        print(f"   Corrected AUC            : {boot_result['auc_corrected']:.3f}")
        print(f"   %95 bootstrap CI         : [{boot_result['ci_low']:.3f}, {boot_result['ci_high']:.3f}]")
        if boot_result['ci_low'] > 0.5:
            print(f"   >>> CI 0.5'i (rastgele tahmin) disliyor -- model gercek ayirt edicilige sahip.")
        else:
            print(f"   >>> CI 0.5'i icine aliyor -- modelin rastgele tahminden anlamli sekilde iyi oldugu kesin degil.")

    print_header("SECTION 2: TEKRARLI STRATIFIED K-FOLD")
    print(f"\n   {N_KFOLD_SPLITS}-fold x {N_KFOLD_REPEATS} tekrar")
    kfold_result = repeated_kfold_single(model_df, FEATURES_RADIOMICS)
    if kfold_result:
        print(f"\n   Out-of-fold AUC (ort +/- sd): {kfold_result['auc_mean']:.3f} +/- {kfold_result['auc_sd']:.3f}")
        print(f"   %95 CI (fold'lar arasi percentile): [{kfold_result['ci_low']:.3f}, {kfold_result['ci_high']:.3f}]")

    print_header("OZET / IKI YONTEMIN KARSILASTIRMASI")
    if boot_result and kfold_result:
        print(f"\n   Bootstrap optimism-corrected AUC : {boot_result['auc_corrected']:.3f}  (CI: [{boot_result['ci_low']:.3f}, {boot_result['ci_high']:.3f}])")
        print(f"   Tekrarli k-fold ortalama AUC      : {kfold_result['auc_mean']:.3f}  (CI: [{kfold_result['ci_low']:.3f}, {kfold_result['ci_high']:.3f}])")
        diff = abs(boot_result['auc_corrected'] - kfold_result['auc_mean'])
        print(f"\n   Iki yontem arasi fark: {diff:.4f} {'(tutarli)' if diff < 0.03 else '(dikkat -- belirgin fark var)'}")
        print(f"\n   NOT: N~411 ve 4 parametre, Faz 6'daki Model A/A' (N~1031) modeline gore")
        print(f"   daha dusuk hasta/parametre orani tasiyor -- daha yuksek optimism beklenebilirdi;")
        print(f"   yukaridaki optimism degeri bu beklentiyle karsilastirilarak yorumlanmalidir.")

    return boot_result, kfold_result


if __name__ == "__main__":
    run_faz6b()
