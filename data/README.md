# Данные

## CWFID (реальные фото с разметкой)

Crop/Weed Field Image Dataset: 60 фото моркови с сорняками (1296×966, RGB),
маски растительности и разметка «культура / сорняк».

1. Скачайте: https://github.com/cwfid/dataset (Code → Download ZIP)
   или `git clone --depth 1 https://github.com/cwfid/dataset.git`
2. Положите содержимое в `data/raw/cwfid/`, должно получиться:

```
data/raw/cwfid/
├── images/        001_image.png ... 060_image.png
├── masks/         001_mask.png ...
├── annotations/   001_annotation.png, 001_annotation.yaml ...
└── train_test_split.yaml
```

Папка `data/raw/` в git не попадает (см. `.gitignore`).

Примечание: в `train_test_split.yaml` номер 28 указан и в train, и в test.
Мы ничего не обучаем, поэтому используем все 60 фото целиком.

## Синтетика

Генерируется кодом (`src/weeddet/synthetic.py`) по seed, в git не хранится.
