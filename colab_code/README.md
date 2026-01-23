# Eagle3 Qwen3-VL Training in Google Colab

Полное решение для обучения Eagle3 draft модели для Qwen3-VL-30B-A3B в Google Colab (offline режим).

## Что это?

Eagle3 - это метод speculative decoding для ускорения inference больших VLM моделей. Вместо генерации одного токена за раз, draft модель предсказывает несколько следующих токенов, которые затем верифицируются target моделью. Это ускоряет генерацию в 2-3 раза без потери качества.

## Требования

- **GPU:**
  - Для генерации ответов: **A100 80GB** (Qwen3-VL-30B) или A100 40GB (Qwen3-VL-4B для тестов)
  - Для генерации hidden states: **A100 80GB** (30B) или A100 40GB (4B)
  - Для обучения draft модели: **A100 40GB** (offline режим экономит память!)
- **Google Colab:** Pro+ рекомендуется
- **Google Drive:** минимум 900GB свободного места
  - Модели: ~60GB
  - Датасеты: ~100GB
  - Hidden states: ~800GB (для 50K samples)
  - Checkpoints: ~100GB
- **Wandb account** (для мониторинга обучения)

## Процесс обучения (Offline режим)

Offline режим разделяет обучение на 3 этапа:

### Этап 0: Подготовка среды (1-2 часа, CPU only)

```python
# В Colab notebook:
from google.colab import drive, userdata
drive.mount('/content/drive')

# Установка зависимостей
!pip install -q angelslim wandb datasets pillow transformers accelerate deepspeed

# Wandb setup
import wandb
wandb.login(key=userdata.get('WANDB_API_KEY'))
```

### Этап 1: Подготовка датасетов (3-5 часов, CPU only)

Запустите `prepare_datasets.ipynb`:
- Загружает ShareGPT4V (English) - 50K samples
- Загружает InternVL/M3IT (Chinese) - 50K samples
- Скачивает изображения (<5MB фильтр)
- Валидирует все изображения
- Сохраняет в Drive: `/Eagle3_Qwen3VL/datasets/`

### Этап 2: Генерация ответов (8-12 часов, A100 80GB GPU) ⚠️

Запустите `generate_responses.ipynb`:
- Загружает **Qwen3-VL-30B-A3B** (требует 80GB VRAM!)
- Генерирует ответы на изображения (batch inference)
- Сохраняет conversations в ShareGPT формат
- Результат: `data_generated.jsonl` для каждого датасета

💡 **Для тестов с A100 40GB:** Используйте Qwen3-VL-4B вместо 30B в CONFIG

### Этап 3: Генерация  hidden states (8-12 часов, A100 80GB GPU) ⭐ ⚠️

Запустите `generate_hidden_states.ipynb`:
- Загружает **Qwen3-VL-30B-A3B** (требует 80GB VRAM!)
- Forward pass через target model
- Извлекает hidden states для каждого sample
- Сохраняет ~800GB .ckpt файлов в Drive
- **После этого target модель больше НЕ НУЖНА!**

💡 **Для тестов с A100 40GB:** Используйте Qwen3-VL-4B (но draft модель будет для 4B, не 30B)

### Этап 4: Обучение draft модели (12-20 часов, A100 GPU) ⚡

Запустите `eagle3_qwen3vl_training_offline.ipynb`:
- Загружает ТОЛЬКО LM head от Qwen3-VL-30B (~4GB вместо 30GB!)
- Создает draft модель (1 layer)
- Загружает hidden states on-the-fly
- Обучение с DeepSpeed ZeRO-2
- GPU usage: ~10GB вместо 38GB!
- В 2x быстрее чем online режим!

## Структура Google Drive

```
/content/drive/MyDrive/Eagle3_Qwen3VL/
├── models/                          # ~60GB
│   ├── Qwen3-VL-30B-A3B/           # Cached от HuggingFace
│   └── Qwen3-VL-2B/                # Для быстрых тестов
├── datasets/                        # ~100GB
│   ├── sharegpt4v_en/
│   │   ├── data_raw.jsonl          # Questions only
│   │   ├── data_generated.jsonl    # With responses
│   │   └── images/                 # ~40GB
│   └── internvl_zh/
│       ├── data_raw.jsonl
│       ├── data_generated.jsonl
│       └── images/                 # ~40GB
├── hidden_states/                   # ~800GB (ключевой этап offline)
│   ├── train/
│   │   ├── rows_0-5000/
│   │   │   ├── data_0.ckpt
│   │   │   └── ...
│   │   └── rows_5000-10000/
│   └── eval/ (optional)
├── checkpoints/                     # ~100GB
│   └── qwen3-vl-30b-eagle3/
│       ├── checkpoint-500/
│       ├── checkpoint-1000/
│       └── ...
└── logs/
    └── wandb/
```

## Преимущества Offline режима

| Критерий | Online | Offline (✅) |
|----------|--------|--------------|
| **GPU для генерации данных** | Не нужна отдельно | **A100 80GB** ⚠️ |
| **GPU для обучения** | A100 80GB | **A100 40GB** ✅ |
| Скорость обучения | 25-40ч | **12-20ч** ⚡ |
| Гибкость | Нужна регенерация | **Можно перезапускать** |
| Эксперименты | Expensive | **Cheap** |
| Стабильность | High memory pressure | **Low usage** |

## ⚡ Альтернатива для A100 40GB

Если у вас только A100 40GB (не 80GB), используйте **Qwen3-VL-4B** вместо 30B:

**Преимущества:**
- ✅ Помещается в A100 40GB
- ✅ Быстрее генерация (2-3 часа вместо 8-12)
- ✅ Меньше места в Drive (~200GB hidden states вместо 800GB)

**Недостатки:**
- ⚠️ Draft модель обучается для 4B, не для 30B
- ⚠️ Немного ниже качество генерации

**Как использовать:**
В `generate_responses.ipynb` измените:
```python
CONFIG = {
    'model_name': 'Qwen/Qwen3-VL-4B',  # Вместо 30B-A3B
}
```

И используйте соответствующий draft config: `qwen3-vl-4b-eagle3-mrope.json`

## Быстрый старт

1. **Клонируйте репозиторий** в Colab:
   ```python
   !git clone https://github.com/Tencent/AngelSlim.git
   %cd AngelSlim
   !git checkout feat/eagle3-qwen3vl-colab-offline
   ```

2. **Запустите ноутбуки по порядку**:
   - `prepare_datasets.ipynb`
   - `generate_responses.ipynb`
   - `generate_hidden_states.ipynb`
   - `eagle3_qwen3vl_training_offline.ipynb`

3. **Мониторинг через Wandb**:
   - Metrics: `train/loss`, `train/acc_0` (acceptance rate)
   - Ожидаемые значения: loss <1.2, acc_0 >60%

## Troubleshooting

### OOM (Out of Memory) ошибки
- Убедитесь что используете DeepSpeed ZeRO-2 (для offline)
- Уменьшите `batch_size` до 1
- Увеличьте `gradient_accumulation_steps`

### Colab disconnection
- Все ноутбуки поддерживают resume
- Hidden states и checkpoints сохранены в Drive
- Просто перезапустите ноутбук

### Slow dataset download
- Используйте меньше `num_workers` (3-4 вместо 8)
- Фильтруйте изображения агрессивнее (<3MB вместо <5MB)

### Dataset errors
- Запустите валидацию: `downloader.validate_dataset()`
- Проверьте что все изображения открываются
- Пересоздайте проблемные samples

## Total GPU time estimate

- Dataset preparation: 3-5ч CPU
- Response generation: 8-12ч GPU
- Hidden states generation: 8-12ч GPU
- Training: 12-20ч GPU
- **Total:** ~28-44 часа GPU time

## Метрики успеха

После обучения:
- ✅ Loss снизился с ~2.5 до <1.2
- ✅ Acceptance rate (acc_0) >60%
- ✅ Step-wise accuracy убывающая: acc_0 > acc_1 > acc_2

## Дополнительная информация

- [Детальный training guide](../docs/colab_training_guide.md)
- [Оригинальная статья Eagle3](https://arxiv.org/abs/2401.15077)
- [AngelSlim документация](https://github.com/Tencent/AngelSlim)

## Лицензия

Apache 2.0 - см. LICENSE файл
