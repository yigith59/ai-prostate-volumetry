"""
PI-CAI Faz 2b - AI Whole-Gland Segmentasyonu Gorsel QA
=========================================================
Amac: Faz 2'de bulunan AI-hacim/manuel-hacim uyumsuzluklarini (ve baska
supheli vakalari) T2W + AI-maske overlay goruntuleriyle gorsel olarak
kontrol etmek. Sadece 4 vaka degil, mumkun oldugunca genis bir supheli-vaka
havuzu cikarilip (genisletilmis oran esigi + AI-hacim fizyolojik
implausibilite + iki AI algoritmasi arasi vaka-bazli uyumsuzluk + bilinen
hatali vaka), ustune rastgele bir "kontrol ornegi" grubu eklenerek kontact
sheet PNG'leri uretilir.

Not: 1476 hastanin TAMAMINI tek tek gorsel incelemek anlamli bir QA degil
(pratik degil, gozden kacirma riski yuksek) -- bunun yerine hedefli+genis
bir supheli havuzu + rastgele kontrol ornegi kombinasyonu kullanilir. Kac
vakanin nasil secildigi asagida acikca raporlanir.
"""

import os
import warnings

import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
DATASET_ROOT = "/content/drive/MyDrive/Uroloji_Projesi/PICAI_Dataset"
IMAGES_ROOT = os.path.join(DATASET_ROOT, "images")
LABELS_ROOT = os.path.join(DATASET_ROOT, "labels")
CLINICAL_CSV = os.path.join(LABELS_ROOT, "clinical_information", "marksheet.csv")

WHOLE_GLAND_DIRS = {
    "Bosma22b": os.path.join(LABELS_ROOT, "anatomical_delineations", "whole_gland", "AI", "Bosma22b"),
    "Guerbet23": os.path.join(LABELS_ROOT, "anatomical_delineations", "whole_gland", "AI", "Guerbet23"),
}
PRIMARY_AI_SOURCE = "Bosma22b"

OUTPUT_DIR = "/content/drive/MyDrive/Uroloji_Projesi/PICAI_Dataset/QA_outputs/faz2b_volume_qa"

KNOWN_FAULTY_CASES = {"11050_1001070"}

# Genisletilmis flagging esikleri (Faz 2'deki 3x/0.33x esiginden daha hassas)
RATIO_HIGH_THRESHOLD = 3.0    # "yuksek oncelik" sapma
RATIO_MED_THRESHOLD = 2.0     # "orta oncelik" sapma (2x-3x arasi da incelemeye alinir)
MIN_PLAUSIBLE_VOLUME_ML = 15.0   # gercek bir yetiskin prostati bu esigin altinda olmaz
MAX_PLAUSIBLE_VOLUME_ML = 250.0  # bu esigin ustu ciddi BPH disinda beklenmez
INTER_ALGO_DISAGREEMENT_RATIO = 1.5  # Bosma22b vs Guerbet23 vaka-bazli uyumsuzluk (genel CCC yuksek olsa da)

N_RANDOM_CONTROL_SAMPLES = 15  # flag'lenmemis, rastgele "normal" karsilastirma ornegi
CASES_PER_CONTACT_SHEET = 12   # 4x3 grid
RANDOM_SEED = 42

PROGRESS_EVERY = 200


# ============================================================
# UTILITIES (Faz 2 ile ayni -- self-contained tutmak icin tekrarlandi)
# ============================================================
def print_header(title):
    print("\n" + "=" * 80)
    print(f"||  {title}")
    print("=" * 80)


def print_subheader(title):
    print(f"\n{'-' * 70}")
    print(f"  {title}")
    print(f"{'-' * 70}")


def load_and_dedupe_marksheet(csv_path):
    print_header("STEP 0: KLINIK VERI HAZIRLIGI")
    df = pd.read_csv(csv_path)
    df['mri_date_parsed'] = pd.to_datetime(df['mri_date'], format='%Y-%m-%d', errors='coerce')
    df_unique = df.sort_values('mri_date_parsed').groupby('patient_id').first().reset_index()
    df_unique['case_id'] = df_unique['patient_id'].astype(int).astype(str) + '_' + df_unique['study_id'].astype(int).astype(str)
    print(f"\n   {len(df_unique)} benzersiz hasta (hasta basina ilk calisma).")
    return df_unique


def compute_ai_volumes(case_ids, label_dir, source_name):
    print_subheader(f"AI hacim hesaplaniyor: {source_name}")
    volumes = {}
    errors = []
    for i, case_id in enumerate(case_ids):
        if PROGRESS_EVERY and i > 0 and i % PROGRESS_EVERY == 0:
            print(f"      ... {i}/{len(case_ids)} vaka islendi")
        path = os.path.join(label_dir, f"{case_id}.nii.gz")
        if not os.path.exists(path):
            errors.append((case_id, "dosya bulunamadi"))
            continue
        try:
            img = sitk.ReadImage(path)
            spacing = img.GetSpacing()
            arr = sitk.GetArrayFromImage(img)
            voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
            volumes[case_id] = int(np.count_nonzero(arr)) * voxel_vol_mm3 / 1000.0
        except Exception as e:
            errors.append((case_id, str(e)))
    print(f"   Basarili: {len(volumes)}/{len(case_ids)} | Hata: {len(errors)}")
    return volumes, errors


# ============================================================
# STEP 1: FLAGGING -- QA ADAYLARINI BELIRLE
# ============================================================
def build_qa_candidate_list(df):
    print_header("STEP 1: QA ADAYLARININ BELIRLENMESI")

    df = df.copy()
    reasons = {cid: [] for cid in df['case_id']}

    # 1) Bilinen hatali vaka
    for cid in KNOWN_FAULTY_CASES:
        if cid in reasons:
            reasons[cid].append("bilinen_hatali_segmentasyon")

    # 2) Manuel vs AI (primary) oran sapmasi -- genisletilmis esikler
    has_both = df['prostate_volume'].notna() & df[f'ai_volume_{PRIMARY_AI_SOURCE}'].notna()
    ratio = df[f'ai_volume_{PRIMARY_AI_SOURCE}'] / df['prostate_volume']
    for cid, r, both in zip(df['case_id'], ratio, has_both):
        if not both:
            continue
        if r > RATIO_HIGH_THRESHOLD or r < 1 / RATIO_HIGH_THRESHOLD:
            reasons[cid].append(f"yuksek_oran_sapmasi(x{r:.2f})")
        elif r > RATIO_MED_THRESHOLD or r < 1 / RATIO_MED_THRESHOLD:
            reasons[cid].append(f"orta_oran_sapmasi(x{r:.2f})")

    # 3) AI hacminin kendisi fizyolojik olarak implausible (manuel veri olmasa bile yakalar --
    #    ozellikle "kurtarilan kohort"ta manuel karsilastirma YOK, bu kontrol tek guvencemiz)
    ai_vol = df[f'ai_volume_{PRIMARY_AI_SOURCE}']
    implausible = ai_vol.notna() & ((ai_vol < MIN_PLAUSIBLE_VOLUME_ML) | (ai_vol > MAX_PLAUSIBLE_VOLUME_ML))
    for cid, flag in zip(df['case_id'], implausible):
        if flag:
            reasons[cid].append("AI_hacim_implausible")

    # 4) Iki AI algoritmasi vaka-bazli uyumsuz (genel CCC yuksek olsa da tekil hatalari yakalar)
    has_both_ai = df['ai_volume_Bosma22b'].notna() & df['ai_volume_Guerbet23'].notna()
    ratio_ai = df['ai_volume_Bosma22b'] / df['ai_volume_Guerbet23']
    for cid, r, both in zip(df['case_id'], ratio_ai, has_both_ai):
        if not both:
            continue
        if r > INTER_ALGO_DISAGREEMENT_RATIO or r < 1 / INTER_ALGO_DISAGREEMENT_RATIO:
            reasons[cid].append(f"AI_algoritmalari_uyumsuz(x{r:.2f})")

    flagged_ids = [cid for cid, r in reasons.items() if len(r) > 0]
    print(f"\n   Toplam flag'lenen vaka: {len(flagged_ids)}/{len(df)}")

    reason_counts = {}
    for cid in flagged_ids:
        for r in reasons[cid]:
            key = r.split('(')[0]
            reason_counts[key] = reason_counts.get(key, 0) + 1
    print("   Gerekce dagilimi:")
    for k, v in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"   |-- {k}: {v}")

    # 5) Rastgele kontrol ornegi (flag'lenmemis, "normal" gorunen vakalardan) -- kor karsilastirma icin
    rng = np.random.RandomState(RANDOM_SEED)
    non_flagged = [cid for cid in df['case_id'] if cid not in flagged_ids]
    n_control = min(N_RANDOM_CONTROL_SAMPLES, len(non_flagged))
    control_ids = list(rng.choice(non_flagged, size=n_control, replace=False))
    for cid in control_ids:
        reasons[cid].append("rastgele_kontrol_ornegi")

    qa_ids = flagged_ids + control_ids
    print(f"\n   + {n_control} rastgele kontrol ornegi eklendi.")
    print(f"   TOPLAM GORSEL KONTROL EDILECEK VAKA: {len(qa_ids)}/{len(df)} ({100*len(qa_ids)/len(df):.1f}%)")
    print(f"\n   NOT: Tum {len(df)} hastayi tek tek incelemek pratik/anlamli bir QA degil.")
    print(f"   Bunun yerine hedefli+genis bir supheli havuzu + rastgele kontrol ornegi kullanildi.")

    qa_df = df[df['case_id'].isin(qa_ids)].copy()
    qa_df['qa_reasons'] = qa_df['case_id'].map(lambda c: "; ".join(reasons[c]))
    return qa_df


# ============================================================
# STEP 2: GORSELLESTIRME
# ============================================================
def find_best_slice(mask_arr):
    """Maske alaninin en buyuk oldugu axial (z) dilimini bulur; maske bossa ortadaki dilimi doner."""
    areas = mask_arr.sum(axis=(1, 2))
    if areas.max() == 0:
        return mask_arr.shape[0] // 2
    return int(np.argmax(areas))


def load_t2w_and_masks(case_id):
    patient_id = case_id.split('_')[0]
    t2w_path = os.path.join(IMAGES_ROOT, patient_id, f"{case_id}_t2w.nii.gz")
    if not os.path.exists(t2w_path):
        return None, None, None

    t2w_arr = sitk.GetArrayFromImage(sitk.ReadImage(t2w_path))

    mask_bosma = None
    path_b = os.path.join(WHOLE_GLAND_DIRS["Bosma22b"], f"{case_id}.nii.gz")
    if os.path.exists(path_b):
        mask_bosma = sitk.GetArrayFromImage(sitk.ReadImage(path_b))

    mask_guerbet = None
    path_g = os.path.join(WHOLE_GLAND_DIRS["Guerbet23"], f"{case_id}.nii.gz")
    if os.path.exists(path_g):
        mask_guerbet = sitk.GetArrayFromImage(sitk.ReadImage(path_g))

    return t2w_arr, mask_bosma, mask_guerbet


def plot_case_on_axis(ax, case_id, row, t2w_arr, mask_bosma, mask_guerbet):
    ref_mask = mask_bosma if mask_bosma is not None else mask_guerbet
    if t2w_arr is None or ref_mask is None:
        ax.text(0.5, 0.5, f"{case_id}\ngoruntu/maske bulunamadi", ha='center', va='center')
        ax.axis('off')
        return

    z = find_best_slice(ref_mask)
    t2w_slice = t2w_arr[z].astype(float)
    lo, hi = np.percentile(t2w_slice, [1, 99])
    t2w_slice = np.clip((t2w_slice - lo) / max(hi - lo, 1e-6), 0, 1)

    ax.imshow(t2w_slice, cmap='gray')
    if mask_bosma is not None and z < mask_bosma.shape[0]:
        ax.contour(mask_bosma[z], levels=[0.5], colors='red', linewidths=1.2)
    if mask_guerbet is not None and z < mask_guerbet.shape[0]:
        ax.contour(mask_guerbet[z], levels=[0.5], colors='cyan', linewidths=1.0, linestyles='dashed')

    manual_v = row.get('prostate_volume', np.nan)
    ai_b = row.get('ai_volume_Bosma22b', np.nan)
    ai_g = row.get('ai_volume_Guerbet23', np.nan)
    title = (f"{case_id}\nmanuel={manual_v:.1f} AI-B={ai_b:.1f} AI-G={ai_g:.1f} mL\n"
             f"{row['qa_reasons'][:40]}")
    ax.set_title(title, fontsize=7)
    ax.axis('off')


def generate_contact_sheets(qa_df, output_dir):
    print_header("STEP 2: KONTAKT SHEET (T2W + AI MASKE OVERLAY) URETIMI")
    os.makedirs(output_dir, exist_ok=True)

    n_pages = int(np.ceil(len(qa_df) / CASES_PER_CONTACT_SHEET))
    print(f"\n   {len(qa_df)} vaka, {CASES_PER_CONTACT_SHEET} vaka/sayfa -> {n_pages} kontakt sheet uretilecek.")
    print(f"   Kirmizi kontur = {PRIMARY_AI_SOURCE} (birincil), Turkuaz kesikli kontur = Guerbet23 (karsilastirma)")
    print(f"   Cikti: Supplementary Figure S1, panel(ler) A-{chr(ord('A') + n_pages - 1)} (600 dpi TIFF, basliksiz)")

    rows_per_page = 3
    cols_per_page = 4
    saved_paths = []

    for page in range(n_pages):
        panel_letter = chr(ord('A') + page)
        chunk = qa_df.iloc[page * CASES_PER_CONTACT_SHEET:(page + 1) * CASES_PER_CONTACT_SHEET]
        fig, axes = plt.subplots(rows_per_page, cols_per_page, figsize=(16, 12))
        axes = axes.flatten()

        for ax, (_, row) in zip(axes, chunk.iterrows()):
            case_id = row['case_id']
            try:
                t2w_arr, mask_bosma, mask_guerbet = load_t2w_and_masks(case_id)
                plot_case_on_axis(ax, case_id, row, t2w_arr, mask_bosma, mask_guerbet)
            except Exception as e:
                ax.text(0.5, 0.5, f"{case_id}\nHATA: {e}", ha='center', va='center', fontsize=7)
                ax.axis('off')

        for ax in axes[len(chunk):]:
            ax.axis('off')

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"SupplFigureS1_{panel_letter}.tiff")
        plt.savefig(out_path, dpi=600, format='tiff', pil_kwargs={'compression': 'tiff_lzw'}, bbox_inches='tight')
        plt.close(fig)
        saved_paths.append(out_path)
        print(f"   |-- kaydedildi: {out_path}")

    return saved_paths


# ============================================================
# MAIN
# ============================================================
def run_faz2b():
    print("\n" + "#" * 80)
    print("#" + " " * 12 + "PI-CAI FAZ 2b - AI SEGMENTASYON GORSEL QA" + " " * 12 + "#")
    print("#" * 80)

    df = load_and_dedupe_marksheet(CLINICAL_CSV)
    case_ids = df['case_id'].tolist()

    for source_name, label_dir in WHOLE_GLAND_DIRS.items():
        volumes, _ = compute_ai_volumes(case_ids, label_dir, source_name)
        df[f'ai_volume_{source_name}'] = df['case_id'].map(volumes)

    qa_df = build_qa_candidate_list(df)
    qa_csv_path = os.path.join(OUTPUT_DIR, "qa_candidate_list.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    qa_df[['case_id', 'prostate_volume', 'ai_volume_Bosma22b', 'ai_volume_Guerbet23', 'qa_reasons']].to_csv(qa_csv_path, index=False)
    print(f"\n   QA aday listesi kaydedildi: {qa_csv_path}")

    saved_paths = generate_contact_sheets(qa_df, OUTPUT_DIR)

    print_header("OZET")
    print(f"   {len(qa_df)} vaka gorsel olarak kontrole hazirlandi.")
    print(f"   {len(saved_paths)} kontakt sheet PNG olusturuldu: {OUTPUT_DIR}")
    print("   Kontakt sheet'leri inceleyip hangi vakalarin gercekten segmentasyon hatasi")
    print("   (AI'nin yanildigi) vs manuel veri hatasi (10155/11051 gibi) oldugunu not edin.")

    return qa_df, saved_paths


if __name__ == "__main__":
    run_faz2b()
