# Krok 0 — Baseline B na pełnych 45k (15 lotnisk): różnorodność danych zmniejsza lukę

> Model: YOLOv10n, protokół identyczny z poprzednimi (100 epok, imgsz 512, batch 64, seed 42).
> Trening: SYNTETYCZNE pełne 45k (train 38250 / val 6750, 15 lotnisk, ~600k instancji).
> Ewaluacja: realny holdout test (2710 obr, 6812 inst).
> Metryki: `results/per_size/syn45k_to_real_baseline.json`.

## Porównanie: podzbiór 3 lotniska vs pełne 15 lotnisk

| metryka | B na 6460 (Atlanta/Basel/Chicago) | **B na 45k (15 lotnisk)** | poprawa | A: real→real |
|---|---|---|---|---|
| mAP@50 | 0.410 | **0.453** | +10% | 0.980 |
| mAP@50:95 | 0.239 | **0.268** | +12% | 0.813 |
| AP_small | 0.262 | 0.286 | +9% | 0.759 |
| AP_medium | 0.328 | 0.384 | +17% | 0.783 |
| **AP_large** | **0.067** | **0.201** | **+200%** | 0.899 |
| FPS | 150 | 132 | — | 150 |

## Kluczowy statement do raportu

> Samo zwiększenie różnorodności danych syntetycznych (z 3 do 15 lotnisk, z 6460 do
> 45000 obrazów) zmniejsza lukę domenową bez żadnej zmiany metody: mAP@50 rośnie
> z 0.41 do 0.45, a mAP@50:95 z 0.24 do 0.27. Najsilniejszy efekt dotyczy DUŻYCH
> obiektów — AP_large rośnie 3× (0.067 → 0.201). To częściowo dezawuuje wcześniejszy
> "paradoks dużych obiektów": katastrofalny AP_large na podzbiorze 3 lotnisk wynikał
> nie tylko z syntetyczności, ale i z ubogiej różnorodności scen/kontekstów dużych
> samolotów. Przy 15 lotniskach model widzi duże samoloty w wielu konfiguracjach,
> więc reprezentacja "large" lepiej generalizuje na realne dane.

## Wniosek metodologiczny
**Różnorodność danych to pierwszy, "darmowy" sposób zmniejszenia luki** — działa przed
jakąkolwiek augmentacją czy mixed-training. To istotne ustalenie do potoku
intelektualnego: zanim sięgniemy po wymyślne metody (A/B/C/D), sama skala i
różnorodność źródła syntetycznego już pomaga.

**Ale luka wciąż ogromna:** 0.45 vs 0.98 (real). Zostaje ~54% do nadrobienia —
to przestrzeń dla eksperymentów A (fotometria), B (częstotliwości), C (mixed
training), D (skala). Ten wynik (0.453 mAP@50) jest **nowym punktem odniesienia**
dla wszystkich eksperymentów na pełnych danych.

## Reprodukcja
```bash
python3 src/coco_to_yolo.py --domain synthetic --classes aircraft --val-frac 0.15
python3 src/train_yolo.py --data data/yolo/synthetic_aircraft/data.yaml \
    --name syn_baseline_full45k_yolov10n --epochs 100 --batch 64 --imgsz 512 --seed 42 \
    --val-data data/yolo/synthetic_aircraft/data.yaml
python3 src/eval_per_size.py --weights runs/syn_baseline_full45k_yolov10n/weights/best.pt \
    --img-dir data/real/PS-RGB_tiled/PS-RGB_tiled \
    --coco-gt data/real/annotations/instances_test_aircraft.json --name syn45k_to_real_baseline
```
