# Обучение модели на открытых данных (улучшение точек тела)

В браузере сейчас используется **BlazePose** (TensorFlow.js) — это готовая модель Google, её **нельзя дообучить** внутри этого репозитория как обычный PyTorch-чекпоинт.

Чтобы **обучить свою** модель позы на открытых датасетах и получить более устойчивые точки (в т.ч. под вашу разметку из 23 ключей), используется папка **`ml-pipeline/`**: Ultralytics **YOLO11-pose**, предобученные веса на **COCO Keypoints** (открытые веса Ultralytics), дообучение на данных, собранных из **COCO 2017** и при необходимости **CrowdPose**.

## Какие открытые источники задействованы

| Источник | Лицензия / условия | Роль |
|----------|-------------------|------|
| **COCO 2017** (images + `person_keypoints_*.json`) | [CC BY 4.0](https://cocodataset.org/#termsofuse) | Основной массив людей с 17 ключами; скрипт `prepare_dataset.py` переводит их в **23 кастомные точки** (включая синтетические пальцы и торс). |
| **CrowdPose** (опционально) | см. репозиторий [CrowdPose](https://github.com/Jeff-sjtu/CrowdPose) | Плотные сцены, частичные перекрытия — полезно для устойчивости. |
| **Предобучение YOLO** (`yolo11n-pose.pt` и др.) | Ultralytics AGPL-3.0 / веса с открытого релиза | Стартовая модель, уже умеющая позу на COCO. |

## Полный цикл (рекомендуется на GPU)

1. Окружение:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r ml-pipeline/requirements.txt
   ```

2. Скачать COCO, слить, конвертировать в YOLO-pose, аугментация окклюзий:

   ```bash
   python ml-pipeline/scripts/bootstrap_all.py --try-crowdpose
   ```

   **Быстрый эксперимент** (подвыборка кадров, меньше диска и времени):

   ```bash
   python ml-pipeline/scripts/bootstrap_all.py --max-images 8000
   ```

3. Обучение (чем больше `epochs` и размер модели (`yolo11s-pose.pt` / `m`), тем обычно лучше качество, но дольше):

   ```bash
   python ml-pipeline/scripts/train.py ^
     --data ml-pipeline/configs/custom_pose.yaml ^
     --model yolo11n-pose.pt ^
     --imgsz 960 ^
     --epochs 200 ^
     --batch 16 ^
     --device 0
   ```

   На CPU замените `--device 0` на `--device cpu` и уменьшите `--batch` (например 4).

4. Экспорт для сервера (в `ml-pipeline/exports/best.onnx`):

   ```bash
   python ml-pipeline/scripts/export_model.py ^
     --weights ml-pipeline/runs/pose_opensource_v1/weights/best.pt ^
     --out-dir ml-pipeline/exports ^
     --imgsz 960
   ```

5. Сервер уже может вызывать `pose_infer.py` с переменной **`POSE_MODEL_PATH`** (по умолчанию `ml-pipeline/exports/best.onnx`). Интеграция с **живой камерой в браузере** — отдельный шаг: сейчас фронт не подменяет BlazePose на ответ API; для этого нужен обмен кадрами с бэкендом или другой runtime.

## Что изменено в скриптах для «лучшего» обучения

- **`train.py`**: косинусный decay LR (`cos_lr`), сильнее геометрические аугментации (углы, масштаб, shear), дольше mosaic (`close_mosaic`), выше вес **`pose`** loss (тонкая настройка ключевых точек), `warmup_epochs=5`.
- **`prepare_dataset.py`**: флаг **`--max-images`** — обучение на подвыборке открытых данных без полного COCO.
- **`export_model.py`**: копирование ONNX в **`ml-pipeline/exports/best.onnx`** для единого пути, как ожидает сервер.
- **`bootstrap_all.py`**: передача **`--max-images`** в этап `prepare_dataset`.

## Важно

- Качество **пальцев** на COCO ограничено: в исходных аннотациях нет пальцев — они инициализируются от запястий; реальный выигрыш даст **своя разметка** поверх камеры.
- Для максимального качества добавьте **свои кадры** (те же 23 точки в COCO-подобном JSON) и снова запустите merge → prepare → train.
