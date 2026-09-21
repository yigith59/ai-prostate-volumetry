"""
Table S3 Panel E icin gerekli legacy dosyalari Drive'da aramak icin (duzeltilmis).

Onceki arama yanlis dosya adi varsayimi kullandi. Gercek kod incelendi:
- Track B: gercek dosya adi 'faz3_trackb_lesion_features.csv' (alt cizgisiz "trackb"),
  QA_outputs/ altina kaydediliyor -- BU DOSYA VAR OLABILIR, ham lezyon ozellikleri
  (model sonucu degil), bulunursa GG2 vs GG3+ modeli guncel pipeline'la yeniden
  fit edilebilir.
- Track A: run_track_a() fonksiyonu SONUCLARI HICBIR CSV'YE KAYDETMIYOR, sadece
  konsola yazdiriyor -- yani geri donulecek bir kaynak dosya hic var olmadi.
  Bu script Track A icin de arama yapiyor ama sifir sonuc bekleniyor (onceki
  calistirmalarda konsol ciktisi disinda kaydedilmis bir sey yoksa).
"""

import glob
import os

DATASET_ROOT = "/content/drive/MyDrive/Uroloji_Projesi/PICAI_Dataset"

# Gercek/olasi dosya adi kaliplari -- alt cizgili VE alt cizgisiz varyantlar
PATTERNS = [
    "*trackb*", "*track_b*",
    "*tracka*", "*track_a*",
    "*faz3*",
    "*lesion_features*",
    "*radiomics*",
]


def main():
    found = set()
    for pattern in PATTERNS:
        matches = glob.glob(os.path.join(DATASET_ROOT, "**", pattern), recursive=True)
        found.update(matches)

    if not found:
        print(f"Hicbir eslesme bulunamadi ({DATASET_ROOT} altinda).")
        print("Hem Track A hem Track B icin legacy/caveat yaklasimi kullanilmali.")
        return

    print(f"{len(found)} olasi dosya/dizin bulundu:\n")
    for path in sorted(found):
        if os.path.isfile(path):
            size_kb = os.path.getsize(path) / 1024
            print(f"  [FILE] {path}  ({size_kb:.1f} KB)")
        else:
            print(f"  [DIR]  {path}")


if __name__ == "__main__":
    main()
