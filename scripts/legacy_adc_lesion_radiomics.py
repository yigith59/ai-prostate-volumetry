"""
PI-CAI Faz 3 - Radyomik (Iki Ayri Analiz)
=========================================================
Faz 0'da bulunan kritik kisit: negatif vakalarin (1075/1500) lezyon
maskeleri PI-CAI dokumantasyonuna gore HER ZAMAN tamamen bostur (lezyon
"yok" degil, hic delinasyon yapilmamis demek). Bu yuzden lezyon-duzeyi
ozellikler (hacim, min/mean ADC) negatiflerin tamaminda tanimsiz olur ve
mevcut ikili (csPCa var/yok) modele dogrudan eklenemez.

Bu yuzden Faz 3 IKI AYRI, temiz analize bolunmustur:

TRACK A (tum hastalar, N~1450): Lezyona ozel degil, whole-gland (Bosma22b,
  Faz 2'de dogrulanmis) maskesi icindeki ADC heterojenitesi (min-ADC,
  degisim katsayisi) -- Faz 2'nin AI-hacim modeline (Model A') paired
  DeLong ile eklenen bir sonraki nested model asamasi.

TRACK B (sadece csPCa=YES vakalari, N~425): Lezyon hacmi, min/mean ADC,
  max cap gibi ozelliklerle ISUP grade siddetini (GG2 vs GG3+) tahmin
  etmek -- klinik olarak degerli, ayri bir soru (aktif izlem vs agresif
  tedavi karari).

Lezyon maskesi kaynagi: `human_expert/resampled` (T2W-grid, Faz 0'da %0
uyumsuz dogrulandi) once denenir, yoksa `human_expert/Pooch25` (T2W-grid'e
yakin, %1 uyumsuz) kullanilir. `original` KULLANILMAZ (Faz 0: %92 uyumsuz).
Granuler (resampled: 0,2,3,4,5) vs binary (Pooch25: 0,1) deger semantigi
farki, maskeler binarize edilerek (>0 -> 1) es hale getirilir; ISUP grade
etiketleri maskeden degil marksheet.csv'nin kendi case_ISUP kolonundan
alinir (boylece iki kaynagin farkli value-scheme'i sonucu etkilemez).

ADC her zaman T2W/lezyon-maskesi grid'ine resample edilir (Faz 0: T2W-ADC
%100 farkli grid, resample zorunlu). Resample sonrasi maske-ici ADC
degerlerinin ne kadari tam 0 (resampler default-fill) cikiyor kontrol
edilip, esik-asiri vakalar hizalanma hatasi kabul edilip disariya birakilir.
"""

import os
import warnings

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import stats, ndimage
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
DATASET_ROOT = "/content/drive/MyDrive/Uroloji_Projesi/PICAI_Dataset"
IMAGES_ROOT = os.path.join(DATASET_ROOT, "images")
LABELS_ROOT = os.path.join(DATASET_ROOT, "labels")
CLINICAL_CSV = os.path.join(LABELS_ROOT, "clinical_information", "marksheet.csv")

WHOLE_GLAND_DIR = os.path.join(LABELS_ROOT, "anatomical_delineations", "whole_gland", "AI", "Bosma22b")
LESION_DIRS = {
    "resampled": os.path.join(LABELS_ROOT, "csPCa_lesion_delineations", "human_expert", "resampled"),
    "Pooch25": os.path.join(LABELS_ROOT, "csPCa_lesion_delineations", "human_expert", "Pooch25"),
}

# Faz 2b gorsel QA ile dogrulanmis hatali segmentasyon (bez-tabanli ozellikler icin de guvenilmez)
KNOWN_FAULTY_CASES = {"11050_1001070"}

MIN_PLAUSIBLE_GLAND_VOLUME_ML = 5.0
MIN_LESION_VOXELS = 5              # bundan kucuk baglantili bilesenler gurultu sayilir
RESAMPLE_ZERO_FRACTION_THRESHOLD = 0.10  # maske-ici resample-sonrasi %10'dan fazla tam-0 varsa hizalanma hatasi

TRACK_B_OUTPUT_CSV = os.path.join(DATASET_ROOT, "QA_outputs", "faz3_trackb_lesion_features.csv")

PROGRESS_EVERY = 200


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


def bootstrap_ci(y_true, y_score, metric_func, n_bootstrap=1000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    scores = []
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    for _ in range(n_bootstrap):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(metric_func(y_true[idx], y_score[idx]))
    if len(scores) == 0:
        print("   UYARI: bootstrap_ci hicbir resample'da her iki sinifi da bulamadi -- NaN donduruluyor.")
        return np.nan, np.nan, np.nan
    scores = np.array(scores)
    lower = np.percentile(scores, (1 - ci) / 2 * 100)
    upper = np.percentile(scores, (1 + ci) / 2 * 100)
    return np.mean(scores), lower, upper


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


def _fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(y_true, y_pred_1, y_pred_2):
    y_true = np.asarray(y_true)
    order = np.argsort(-y_true)
    m = int(np.sum(y_true[order]))
    preds = np.vstack([np.asarray(y_pred_1), np.asarray(y_pred_2)])[:, order]
    aucs, delongcov = _fast_delong(preds, m)
    l = np.array([[1, -1]])
    var = np.dot(np.dot(l, delongcov), l.T)
    z = np.abs(aucs[0] - aucs[1]) / np.sqrt(var[0, 0])
    p_value = 2 * (1 - stats.norm.cdf(z))
    return aucs[0], aucs[1], p_value


def load_and_dedupe_marksheet(csv_path):
    print_header("STEP 0: KLINIK VERI HAZIRLIGI")
    df = pd.read_csv(csv_path)
    df['mri_date_parsed'] = pd.to_datetime(df['mri_date'], format='%Y-%m-%d', errors='coerce')
    df_unique = df.sort_values('mri_date_parsed').groupby('patient_id').first().reset_index()
    df_unique['case_id'] = df_unique['patient_id'].astype(int).astype(str) + '_' + df_unique['study_id'].astype(int).astype(str)
    df_unique['csPCa_binary'] = (df_unique['case_csPCa'] == 'YES').astype(int)
    print(f"\n   {len(df_unique)} benzersiz hasta (hasta basina ilk calisma).")
    return df_unique


def compute_ai_volumes(case_ids, label_dir):
    volumes = {}
    for i, case_id in enumerate(case_ids):
        if PROGRESS_EVERY and i > 0 and i % PROGRESS_EVERY == 0:
            print(f"      ... {i}/{len(case_ids)} vaka islendi")
        path = os.path.join(label_dir, f"{case_id}.nii.gz")
        if not os.path.exists(path):
            continue
        try:
            img = sitk.ReadImage(path)
            spacing = img.GetSpacing()
            arr = sitk.GetArrayFromImage(img)
            volumes[case_id] = int(np.count_nonzero(arr)) * spacing[0] * spacing[1] * spacing[2] / 1000.0
        except Exception:
            continue
    return volumes


def resample_to_reference(moving_img, reference_img, interpolator=sitk.sitkLinear):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_img)
    resampler.SetInterpolator(interpolator)
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0.0)
    return resampler.Execute(moving_img)


def find_lesion_mask_path(case_id):
    """Once resampled, yoksa Pooch25. 'original' hic kullanilmiyor (Faz 0: T2W'ye gore %92 uyumsuz)."""
    for source in ("resampled", "Pooch25"):
        path = os.path.join(LESION_DIRS[source], f"{case_id}.nii.gz")
        if os.path.exists(path):
            return path, source
    return None, None


# ============================================================
# TRACK A: WHOLE-GLAND ADC HETEROJENITESI (TUM HASTALAR)
# ============================================================
def extract_gland_adc_features(case_id, gland_mask_path):
    patient_id = case_id.split('_')[0]
    t2w_path = os.path.join(IMAGES_ROOT, patient_id, f"{case_id}_t2w.nii.gz")
    adc_path = os.path.join(IMAGES_ROOT, patient_id, f"{case_id}_adc.nii.gz")
    if not (os.path.exists(t2w_path) and os.path.exists(adc_path) and os.path.exists(gland_mask_path)):
        return None

    t2w_img = sitk.ReadImage(t2w_path)
    adc_img = sitk.ReadImage(adc_path)
    gland_mask_img = sitk.ReadImage(gland_mask_path)

    adc_resampled = resample_to_reference(adc_img, t2w_img, sitk.sitkLinear)
    adc_arr = sitk.GetArrayFromImage(adc_resampled)
    gland_arr = sitk.GetArrayFromImage(gland_mask_img) > 0

    if gland_arr.sum() == 0:
        return None

    adc_in_gland = adc_arr[gland_arr]
    zero_fraction = float(np.mean(adc_in_gland == 0))
    if zero_fraction > RESAMPLE_ZERO_FRACTION_THRESHOLD:
        return {"case_id": case_id, "resample_failed": True, "zero_fraction": zero_fraction}

    nonzero_vals = adc_in_gland[adc_in_gland != 0]
    if len(nonzero_vals) < 10:
        return None

    return {
        "case_id": case_id,
        "resample_failed": False,
        "zero_fraction": zero_fraction,
        "gland_adc_p5": float(np.percentile(nonzero_vals, 5)),
        "gland_adc_mean": float(np.mean(nonzero_vals)),
        "gland_adc_cv": float(np.std(nonzero_vals) / np.mean(nonzero_vals)) if np.mean(nonzero_vals) != 0 else np.nan,
    }


def run_track_a(df):
    print_header("TRACK A: WHOLE-GLAND ADC HETEROJENITESI (TUM HASTALAR)")

    df = df.copy()
    case_ids = df['case_id'].tolist()

    print_subheader("A.0 AI Whole-Gland Hacmi (Faz 2 ile tutarli)")
    volumes = compute_ai_volumes(case_ids, WHOLE_GLAND_DIR)
    df['ai_volume_primary'] = df['case_id'].map(volumes)
    implausible = df['ai_volume_primary'].notna() & (df['ai_volume_primary'] < MIN_PLAUSIBLE_GLAND_VOLUME_ML)
    df.loc[implausible, 'ai_volume_primary'] = np.nan
    faulty = df['case_id'].isin(KNOWN_FAULTY_CASES)
    df.loc[faulty, 'ai_volume_primary'] = np.nan

    print_subheader("A.1 Bez-Ici ADC Ozellikleri Cikariliyor")
    records = []
    n_resample_failed = 0
    for i, row in df.iterrows():
        if PROGRESS_EVERY and i > 0 and i % PROGRESS_EVERY == 0:
            print(f"      ... {i}/{len(df)} vaka islendi")
        if pd.isna(row['ai_volume_primary']):
            continue
        gland_mask_path = os.path.join(WHOLE_GLAND_DIR, f"{row['case_id']}.nii.gz")
        try:
            feats = extract_gland_adc_features(row['case_id'], gland_mask_path)
        except Exception as e:
            feats = None
        if feats is None:
            continue
        if feats.get("resample_failed"):
            n_resample_failed += 1
            continue
        records.append(feats)

    print(f"\n   Basarili ADC ozellik cikarimi: {len(records)}/{len(df)}")
    print(f"   Resample hizalanma hatasi nedeniyle atlanan: {n_resample_failed} "
          f"(gland-ici resample-ADC degerlerinin >{RESAMPLE_ZERO_FRACTION_THRESHOLD*100:.0f}%'i tam 0 cikti)")

    if len(records) == 0:
        print("   UYARI: Hicbir vakada ADC ozelligi cikarilamadi -- Track A model karsilastirmasi atlaniyor.")
        for col in ['gland_adc_p5', 'gland_adc_mean', 'gland_adc_cv']:
            df[col] = np.nan
        return df

    feat_df = pd.DataFrame(records, columns=['case_id', 'resample_failed', 'zero_fraction',
                                              'gland_adc_p5', 'gland_adc_mean', 'gland_adc_cv'])
    df = df.merge(feat_df, on='case_id', how='left')

    print_subheader("A.2 Nested Model: Model A' (AI-hacim) vs Model A'' (+ ADC heterojenite)")
    complete = df.dropna(subset=['patient_age', 'psa', 'ai_volume_primary', 'gland_adc_p5', 'gland_adc_cv', 'center', 'csPCa_binary']).copy()
    complete = complete[complete['center'] != 'PCNN'].copy()
    print(f"\n   Tam veri kohortu (PCNN haric): N = {len(complete)}")

    if len(complete) < 30 or complete['csPCa_binary'].nunique() < 2:
        print("   Yetersiz N/sinif cesitliligi, model karsilastirmasi atlaniyor.")
        return df

    complete['age_10yr'] = complete['patient_age'] / 10.0
    complete['psa_log'] = np.log1p(complete['psa'])
    complete['vol_log_ai'] = np.log1p(complete['ai_volume_primary'])
    complete['center_RUMC'] = (complete['center'] == 'RUMC').astype(int)

    y = complete['csPCa_binary'].astype(int)

    adc_feat_corr = complete[['gland_adc_p5', 'gland_adc_cv']].corr().iloc[0, 1]
    print(f"\n   gland_adc_p5 vs gland_adc_cv korelasyonu: r={adc_feat_corr:.3f}")
    if abs(adc_feat_corr) > 0.9:
        print(f"   UYARI: |r|>0.9 -- iki ozellik neredeyse ayni bilgiyi tasiyor, model tekillesebilir. "
              f"Sadece gland_adc_p5 ile devam ediliyor.")
        adc_features = ['gland_adc_p5']
    else:
        adc_features = ['gland_adc_p5', 'gland_adc_cv']

    try:
        X1 = sm.add_constant(complete[['age_10yr', 'psa_log', 'vol_log_ai', 'center_RUMC']])
        model1 = sm.Logit(y, X1).fit(disp=0)
        pred1 = model1.predict(X1)

        X2 = sm.add_constant(complete[['age_10yr', 'psa_log', 'vol_log_ai', 'center_RUMC'] + adc_features])
        model2 = sm.Logit(y, X2).fit(disp=0)
        pred2 = model2.predict(X2)
    except np.linalg.LinAlgError as e:
        print(f"   UYARI: Model fit basarisiz (tekil matris / olasi coklu-dogrusallik): {e}")
        print(f"   Track A model karsilastirmasi atlaniyor -- ozellik setini gozden gecirin.")
        return df

    print(f"\n   Model A'  (AI-hacim)              -- AIC: {model1.aic:.1f}, Pseudo R2: {model1.prsquared:.3f}")
    print(f"   Model A'' (+ ADC heterojenite)     -- AIC: {model2.aic:.1f}, Pseudo R2: {model2.prsquared:.3f}")

    auc1, auc2, p_delong = delong_roc_test(y.values, pred1.values, pred2.values)
    print(f"\n   Model A'  in-sample AUC: {auc1:.3f}")
    print(f"   Model A'' in-sample AUC: {auc2:.3f}")
    print(f"   DeLong paired test: p = {p_delong:.4f} {'(anlamli fark)' if p_delong < 0.05 else '(anlamli fark yok)'}")
    print(f"\n   NOT: In-sample sonuclar, Faz 6'da optimism-corrected dogrulama gerekiyor.")

    return df


# ============================================================
# TRACK B: LEZYON RADYOMIGI -- SADECE csPCa=YES (ISUP GRADE SIDDETI)
# ============================================================
def extract_lesion_features(case_id, lesion_mask_path, gland_mask_path):
    patient_id = case_id.split('_')[0]
    t2w_path = os.path.join(IMAGES_ROOT, patient_id, f"{case_id}_t2w.nii.gz")
    adc_path = os.path.join(IMAGES_ROOT, patient_id, f"{case_id}_adc.nii.gz")
    if not (os.path.exists(t2w_path) and os.path.exists(adc_path) and os.path.exists(gland_mask_path)):
        return {"case_id": case_id, "skip_reason": "eksik_goruntu_dosyasi"}

    t2w_img = sitk.ReadImage(t2w_path)
    spacing = t2w_img.GetSpacing()
    voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]

    lesion_arr_raw = sitk.GetArrayFromImage(sitk.ReadImage(lesion_mask_path))
    lesion_binary = (lesion_arr_raw > 0)
    if lesion_binary.sum() == 0:
        return {"case_id": case_id, "skip_reason": "bos_lezyon_maskesi"}

    labeled, n_components_raw = ndimage.label(lesion_binary)
    sizes = ndimage.sum(lesion_binary, labeled, range(1, n_components_raw + 1))
    valid_labels = [i + 1 for i, s in enumerate(sizes) if s >= MIN_LESION_VOXELS]
    if len(valid_labels) == 0:
        return {"case_id": case_id, "skip_reason": "sadece_gurultu_bileseni"}

    index_label = valid_labels[int(np.argmax([sizes[l - 1] for l in valid_labels]))]
    index_mask = (labeled == index_label)
    n_voxels = int(index_mask.sum())
    lesion_volume_mL = n_voxels * voxel_vol_mm3 / 1000.0

    coords = np.argwhere(index_mask)
    extent_vox = coords.max(axis=0) - coords.min(axis=0) + 1
    extent_mm = extent_vox * np.array(spacing[::-1])  # array is z,y,x; spacing is x,y,z
    max_diameter_mm = float(np.max(extent_mm))

    adc_img = sitk.ReadImage(adc_path)
    adc_resampled = resample_to_reference(adc_img, t2w_img, sitk.sitkLinear)
    adc_arr = sitk.GetArrayFromImage(adc_resampled)

    adc_in_lesion = adc_arr[index_mask]
    zero_fraction = float(np.mean(adc_in_lesion == 0)) if len(adc_in_lesion) > 0 else 1.0
    if zero_fraction > RESAMPLE_ZERO_FRACTION_THRESHOLD:
        return {"case_id": case_id, "resample_failed": True}

    nonzero_lesion_adc = adc_in_lesion[adc_in_lesion != 0]
    if len(nonzero_lesion_adc) < 5:
        return {"case_id": case_id, "skip_reason": "yetersiz_adc_voksel"}

    gland_arr = sitk.GetArrayFromImage(sitk.ReadImage(gland_mask_path)) > 0
    # Sadece index lezyonu degil, gurultu-esigini gecen TUM lezyon bilesenlerini (multifokal
    # vakalarda ikincil lezyonlar dahil) referans/normal doku havuzundan cikar.
    all_valid_lesions_mask = np.isin(labeled, valid_labels)
    reference_mask = gland_arr & (~all_valid_lesions_mask)
    adc_in_reference = adc_arr[reference_mask]
    adc_in_reference = adc_in_reference[adc_in_reference != 0]

    adc_ratio = np.nan
    if len(adc_in_reference) >= 20:
        ref_median = np.median(adc_in_reference)
        if ref_median != 0:
            adc_ratio = float(np.mean(nonzero_lesion_adc) / ref_median)

    return {
        "case_id": case_id,
        "resample_failed": False,
        "lesion_source": None,  # set by caller
        "lesion_volume_mL": lesion_volume_mL,
        "lesion_count": len(valid_labels),
        "max_diameter_mm": max_diameter_mm,
        "lesion_adc_min": float(np.min(nonzero_lesion_adc)),
        "lesion_adc_mean": float(np.mean(nonzero_lesion_adc)),
        "lesion_adc_ratio": adc_ratio,
    }


def run_track_b(df):
    print_header("TRACK B: LEZYON RADYOMIGI -- ISUP GRADE SIDDETI (SADECE csPCa=YES)")

    positive_df = df[(df['case_csPCa'] == 'YES') & (~df['case_id'].isin(KNOWN_FAULTY_CASES))].copy()
    print(f"\n   csPCa=YES vaka sayisi: {len(positive_df)}")

    records = []
    n_no_mask, source_counts = 0, {"resampled": 0, "Pooch25": 0}
    skip_reason_counts = {}
    for pos, (_, row) in enumerate(positive_df.iterrows()):
        if PROGRESS_EVERY and pos > 0 and pos % PROGRESS_EVERY == 0:
            print(f"      ... {pos}/{len(positive_df)} vaka islendi")

        lesion_path, source = find_lesion_mask_path(row['case_id'])
        if lesion_path is None:
            n_no_mask += 1
            continue

        gland_mask_path = os.path.join(WHOLE_GLAND_DIR, f"{row['case_id']}.nii.gz")
        try:
            feats = extract_lesion_features(row['case_id'], lesion_path, gland_mask_path)
        except Exception as e:
            feats = {"case_id": row['case_id'], "skip_reason": f"exception: {e}"}

        if feats.get("resample_failed"):
            skip_reason_counts["resample_hizalanma_hatasi"] = skip_reason_counts.get("resample_hizalanma_hatasi", 0) + 1
            continue
        if "skip_reason" in feats:
            reason = feats["skip_reason"].split(":")[0]
            skip_reason_counts[reason] = skip_reason_counts.get(reason, 0) + 1
            continue

        feats["lesion_source"] = source
        source_counts[source] += 1
        records.append(feats)

    print(f"\n   Lezyon maskesi bulunamayan: {n_no_mask}")
    print(f"   Basarili ozellik cikarimi: {len(records)} (kaynak: resampled={source_counts['resampled']}, Pooch25={source_counts['Pooch25']})")
    if skip_reason_counts:
        print(f"   Diger nedenlerle atlanan (toplam {sum(skip_reason_counts.values())}):")
        for reason, count in sorted(skip_reason_counts.items(), key=lambda x: -x[1]):
            print(f"   |-- {reason}: {count}")
    n_accounted = n_no_mask + len(records) + sum(skip_reason_counts.values())
    if n_accounted != len(positive_df):
        print(f"   UYARI: {len(positive_df)} vakadan sadece {n_accounted} tanesi hesaba katildi -- fark kontrol edilmeli.")

    if len(records) < 30:
        print("   Yetersiz N, Track B analizi atlaniyor.")
        return None

    feat_df = pd.DataFrame(records)
    merged = positive_df.merge(feat_df, on='case_id', how='inner')
    merged['high_grade'] = (merged['case_ISUP'] >= 3).astype(int)

    os.makedirs(os.path.dirname(TRACK_B_OUTPUT_CSV), exist_ok=True)
    merged.to_csv(TRACK_B_OUTPUT_CSV, index=False)
    print(f"\n   Track B lezyon ozellikleri kaydedildi: {TRACK_B_OUTPUT_CSV}")

    print_subheader("B.1 Anatomik Tutarlilik: Lezyon Boyutu ve Grade Dagilimi")
    print(f"\n   GG2 (case_ISUP=2): {(merged['case_ISUP']==2).sum()}")
    print(f"   GG3+ (case_ISUP>=3): {(merged['case_ISUP']>=3).sum()}")
    print(f"\n   Medyan lezyon hacmi -- GG2: {merged.loc[merged['high_grade']==0,'lesion_volume_mL'].median():.2f} mL, "
          f"GG3+: {merged.loc[merged['high_grade']==1,'lesion_volume_mL'].median():.2f} mL")

    print_subheader("B.2 Univariate Karsilastirma (GG2 vs GG3+)")
    for var in ['lesion_volume_mL', 'lesion_adc_min', 'lesion_adc_mean', 'lesion_adc_ratio', 'max_diameter_mm']:
        g2 = merged.loc[merged['high_grade'] == 0, var].dropna()
        g3 = merged.loc[merged['high_grade'] == 1, var].dropna()
        if len(g2) < 5 or len(g3) < 5:
            continue
        stat, p = stats.mannwhitneyu(g2, g3, alternative='two-sided')
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"   {var:<18}: GG2 medyan={g2.median():.2f}, GG3+ medyan={g3.median():.2f}, p={p:.4f} {sig}")

    print_subheader("B.3 Multivariable Model: Radyomik ile GG3+ Tahmini")
    model_df = merged.dropna(subset=['lesion_volume_mL', 'lesion_adc_min', 'lesion_adc_ratio', 'patient_age']).copy()
    print(f"\n   Model kohortu: N = {len(model_df)}")

    if len(model_df) < 30 or model_df['high_grade'].nunique() < 2:
        print("   Yetersiz N/sinif cesitliligi, multivariable model atlaniyor.")
        return merged

    model_df['age_10yr'] = model_df['patient_age'] / 10.0
    model_df['lesion_vol_log'] = np.log1p(model_df['lesion_volume_mL'])

    y = model_df['high_grade'].astype(int)
    X = sm.add_constant(model_df[['age_10yr', 'lesion_vol_log', 'lesion_adc_min', 'lesion_adc_ratio']])
    model = sm.Logit(y, X).fit(disp=0)
    pred = model.predict(X)
    auc = roc_auc_score(y, pred)
    _, auc_l, auc_u = bootstrap_ci(y.values, pred.values, roc_auc_score)

    var_names = ['Intercept', 'Age (per 10 yrs)', 'Lezyon hacmi (log)', 'Lezyon min-ADC', 'Lezyon ADC-orani']
    print(f"\n   {'Variable':<20} | {'Coef':^10} | {'OR':^10} | {'p-value':^10}")
    for i, var in enumerate(var_names):
        coef = model.params.iloc[i]
        p_val = model.pvalues.iloc[i]
        or_val = np.exp(coef) if var != 'Intercept' else np.nan
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"   {var:<20} | {coef:^10.3f} | {or_val:^10.2f} | {p_val:^10.4f} {sig}")

    print(f"\n   GG3+ tahmini in-sample AUC: {auc:.3f} (95% CI {auc_l:.3f}-{auc_u:.3f})")
    print(f"   NOT: In-sample sonuc, Faz 6'da optimism-corrected dogrulama gerekiyor.")
    if auc > 0.999 or (np.abs(model.params) > 20).any():
        print(f"   UYARI: AUC~1.0 ve/veya asiri buyuk katsayi -- olasi 'perfect/quasi-complete separation' "
              f"belirtisi. OR degerleri bu durumda anlamsiz derecede buyuk cikar ve guvenilmezdir. "
              f"Gercek veride (N~425) beklenmez, ama katsayilari yorumlamadan once mutlaka kontrol edin "
              f"(orn. statsmodels yerine Firth'in penalized lojistik regresyonu dusunulebilir).")

    return merged


# ============================================================
# MAIN
# ============================================================
def run_faz3():
    print("\n" + "#" * 80)
    print("#" + " " * 20 + "PI-CAI FAZ 3 - RADYOMIK (2 TRACK)" + " " * 20 + "#")
    print("#" * 80)

    df = load_and_dedupe_marksheet(CLINICAL_CSV)

    df_a = run_track_a(df)
    merged_b = run_track_b(df)

    print_header("OZET")
    print("   Faz 3 tamamlandi (Track A: tum kohort ADC heterojenitesi, Track B: GG2 vs GG3+ radyomik).")

    return df_a, merged_b


if __name__ == "__main__":
    run_faz3()
