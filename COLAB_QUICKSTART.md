# Eagle3 Qwen3-VL Colab Training - Quick Start

## 🎯 Цель

Обучить Eagle3 draft модель для Qwen3-VL-30B-A3B в Google Colab (offline режим).

## ⚡ Команды для клонирования

### В Google Colab (в начале каждого ноутбука):

```python
# Замените YOUR_USERNAME на ваш GitHub username!
!git clone -b feat/eagle3-qwen3vl-colab-offline \
    https://github.com/YOUR_USERNAME/AngelSlim.git /content/AngelSlim
%cd /content/AngelSlim
```

### Локально (один раз, чтобы запушить ветку):

```bash
# 1. Добавьте ваш форк (замените YOUR_USERNAME!)
git remote add myfork https://github.com/YOUR_USERNAME/AngelSlim.git

# 2. Запушьте ветку
git push -u myfork feat/eagle3-qwen3vl-colab-offline

# 3. Проверьте
git remote -v
```

---

## 📋 Полный Pipeline

### Этап 0: Подготовка (один раз)

**Время:** 5-10 минут

1. **Создайте Colab secrets:**
   - Откройте любой Colab ноутбук
   - Слева: 🔑 Secrets → Add new secret
   - Name: `WANDB_API_KEY`
   - Value: ваш API key с https://wandb.ai/settings

2. **Настройте Google Drive:**
   - Минимум 900GB свободного места (для 100K samples)
   - Или 200GB для тестов (200-1000 samples)

### Этап 1: Подготовка датасетов

**Файл:** `colab_code/prepare_datasets.ipynb`
**Время:** 3-5 часов (CPU only)
**GPU:** Не требуется

**Для быстрого теста:**
```python
CONFIG = {
    'english_dataset': {
        'num_samples': 200,  # Вместо 50000
    },
    'chinese_dataset': {
        'num_samples': 200,  # Вместо 50000
    },
}
```

**Результат:** `data_raw.jsonl` + изображения в Drive

---

### Этап 2: Генерация ответов

**Файл:** `colab_code/generate_responses.ipynb`
**Время:** 8-12 часов для 100K samples (или 1-2 часа для 400)
**GPU:**
- **A100 80GB** для Qwen3-VL-30B-A3B
- **A100 40GB** для Qwen3-VL-4B (альтернатива)

**Для A100 40GB:**
```python
CONFIG = {
    'model_name': 'Qwen/Qwen3-VL-4B',  # Вместо 30B-A3B
}
```

**Результат:** `data_generated.jsonl` с conversations

---

### Этап 3: Генерация Hidden States ⭐

**Файл:** `colab_code/generate_hidden_states.ipynb`
**Время:** 8-12 часов для 100K samples
**GPU:**
- **A100 80GB** для Qwen3-VL-30B
- **A100 40GB** для Qwen3-VL-4B

**Важно:** Используйте ту же модель, что и в этапе 2!

**Результат:** ~800GB .ckpt файлов (или ~200GB для 4B)

**После этого этапа target модель больше НЕ НУЖНА!**

---

### Этап 4: Обучение Draft Модели ⚡

**Файл:** `colab_code/eagle3_qwen3vl_training_offline.ipynb`
**Время:** 12-20 часов для 100K samples
**GPU:** **A100 40GB** достаточно! (только ~10GB используется)

**Конфигурация:**
```python
CONFIG = {
    'target_model_name': 'Qwen/Qwen3-VL-30B-A3B',  # Только LM head
    'draft_config': 'angelslim/compressor/speculative/train/configs/qwen3-vl-30b-a3b-eagle3-mrope.json',
    'num_train_epochs': 3,
    'gradient_accumulation_steps': 8,
}
```

**Результат:** Обученная draft модель в checkpoints/

---

## 🧪 Быстрый тест (200 samples)

### Для проверки всего pipeline за ~3-4 часа:

**1. prepare_datasets.ipynb:**
```python
'num_samples': 100  # EN
'num_samples': 100  # ZH
```

**2. generate_responses.ipynb:**
```python
'model_name': 'Qwen/Qwen3-VL-2B'  # Самая быстрая
```

**3. generate_hidden_states.ipynb:**
```python
'model_name': 'Qwen/Qwen3-VL-2B'  # Та же модель!
```

**4. eagle3_qwen3vl_training_offline.ipynb:**
```python
QUICK_TEST = True  # Включить быстрый тест
'num_train_epochs': 1
```

**Итого:** ~3-4 часа от начала до конца

---

## 📊 GPU Требования (обновлено!)

| Этап | Модель | GPU |
|------|--------|-----|
| 1. Датасеты | - | CPU only ✅ |
| 2. Генерация ответов | 30B-A3B | **A100 80GB** ⚠️ |
| 2. Генерация ответов | 4B | A100 40GB ✅ |
| 3. Hidden states | 30B-A3B | **A100 80GB** ⚠️ |
| 3. Hidden states | 4B | A100 40GB ✅ |
| 4. **Обучение** | Draft | **A100 40GB** ✅ |

**Итог:**
- **Production (30B):** Нужна A100 80GB для этапов 2-3, затем A100 40GB для обучения
- **Testing (4B):** A100 40GB достаточно для всех этапов!

---

## ✅ Success Criteria

После обучения проверьте метрики:

```python
# В последней ячейке eagle3_qwen3vl_training_offline.ipynb
✅ train/loss < 1.2           # Должен снизиться с ~2.5
✅ train/acc_0 > 0.6          # Acceptance rate >60%
✅ train/acc_0 > train/acc_1   # Убывающая точность
```

---

## 🐛 Troubleshooting

### OOM (Out of Memory)
```python
# Уменьшите batch size
'per_device_train_batch_size': 1,
'gradient_accumulation_steps': 16,  # Было 8
```

### Colab disconnection
- Все ноутбуки поддерживают resume
- Просто перезапустите с того же места

### Slow downloads
```python
# Уменьшите workers
'num_workers': 2,  # Было 4
```

### Dataset errors
```python
# Запустите валидацию
downloader.validate_dataset()
```

---

## 📁 Итоговая структура Google Drive

```
/content/drive/MyDrive/Eagle3_Qwen3VL/
├── models/                        # ~60GB
│   ├── Qwen3-VL-30B-A3B/
│   └── Qwen3-VL-4B/
├── datasets/                      # ~100GB
│   ├── sharegpt4v_en/
│   │   ├── data_raw.jsonl
│   │   ├── data_generated.jsonl
│   │   └── images/
│   └── m3it_zh/
│       ├── data_raw.jsonl
│       ├── data_generated.jsonl
│       └── images/
├── hidden_states/                 # ~800GB
│   └── train/
│       ├── rows_0-5000/
│       └── rows_5000-10000/
└── checkpoints/                   # ~100GB
    └── qwen3-vl-30b-eagle3/
        ├── checkpoint-500/
        └── checkpoint-1000/
```

---

## 🚀 Следующие шаги после обучения

1. **Тестирование:**
   - Запустите inference с draft моделью
   - Измерьте speedup vs baseline

2. **Оптимизация:**
   - Попробуйте разные hyperparameters
   - Добавьте больше данных

3. **Deployment:**
   - Экспортируйте draft модель
   - Интегрируйте с vLLM

---

## 📚 Дополнительная документация

- Детальный guide: `colab_code/README.md`
- Описание loss_mask: Создается автоматически при обучении
- Wandb dashboard: https://wandb.ai/

---

## ⏱️ Итоговые оценки времени

### Production (100K samples, 30B):
- Этап 1: 3-5ч CPU
- Этап 2: 10-12ч A100-80GB
- Этап 3: 10-12ч A100-80GB
- Этап 4: 15-20ч A100-40GB
- **Total: ~38-49 часов GPU**

### Quick Test (200 samples, 2B):
- Этап 1: 30 мин CPU
- Этап 2: 30 мин A100-40GB
- Этап 3: 1ч A100-40GB
- Этап 4: 1-2ч A100-40GB
- **Total: ~3-4 часа GPU** ⚡
