# Исследовательский код: YOLO + EfficientSAM3

Этот репозиторий содержит Python-реализацию исследовательского пайплайна для детекции и сегментации объектов в сложных условиях видимости.

Основа реализации и логика экспериментов соответствуют материалам исследования (см. `Hybrid_YOLO_EfficientSAM3.pdf`) и исходным ноутбукам в `drafts/`.

## Связь со статьей
Для описания методов, постановки задачи и интерпретации результатов используйте статью:
- `Hybrid_YOLO_EfficientSAM3.pdf` (рукопись исследования)

Также используются первоисточники моделей:
- EfficientSAM3: <https://github.com/SimonZeng7108/efficientsam3>
- YOLO (Ultralytics): <https://github.com/ultralytics/ultralytics>

## Структура репозитория
```text
github_publication_code/
├── images/                          # результаты сегментации изображений в ходе экспериментов
├── research/                        # исходные ipynb-файлы с исследованием
│   ├── SAM3_Research.ipynb
│   ├── Yolo_SAM3_Research.ipynb
│   └── Yolo_SAM3_improve.ipynb
├── src/                             # основной код
│   ├── __init__.py
│   ├── cli.py
│   ├── constants.py
│   ├── metrics.py
│   ├── pipeline.py
│   └── visibility.py
├── requirements.txt
└── pyproject.toml
```

## Что реализовано в коде
- пайплайн `YOLO + EfficientSAM3`;
- ансамбль YOLO с детектором мелких объектов;
- опциональный visibility-aware препроцессинг (`night/fog/rain/snow`);
- специализированное слияние боксов с весами;
- адаптивные пороги уверенности;
- асширение боксов для мелких классов;
- кластеризация боксов по IoU;
- Mask NMS (подавление немаксимумов для масок);
- класс-специфичные пороги IoU для масок;
- расчёт метрик `IoU/mIoU` (instance-level и class-level) для BDD100K;
- Memory management для стабильности;
- пакетный запуск эксперимента через CLI.

## Инструкция по запуску

### 1. Подготовка окружения
```bash
cd github_publication_code
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 2. Установка EfficientSAM3
```bash
git clone https://github.com/SimonZeng7108/efficientsam3.git
pip install -e efficientsam3
```

### 3. Подготовка данных и весов
Необходимо указать:
- путь к изображениям BDD100K (например, `images/val`);
- путь к маскам BDD100K (например, `labels/val`);
- путь к чекпоинту EfficientSAM3 (`.pt`);
- путь/имя весов YOLO (например, `yolov8n.pt`).

### 4. Запуск оценки
```bash
PYTHONPATH=src python -m yolo_efficientsam3.cli \
  --images /path/to/bdd100k/images/val \
  --masks /path/to/bdd100k/labels/val \
  --sam-checkpoint /path/to/efficient_sam3_checkpoint.pt \
  --yolo-model yolov8n.pt \
  --limit 100 \
  --preprocess \
  --output metrics.json
```

После выполнения будет создан файл `metrics.json` с итоговыми метриками и временными характеристиками.

## Примечания
- Рекомендуемый протокол экспериментов: `BDD100K + Google Colab T4`.
- Конвертация EfficientSAM3 в TensorRT в текущей версии репозитория не включена (архитектурные/операторные ограничения).
