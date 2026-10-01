# Локальная модель имён

`cyrillic_PP-OCRv3_rec_mobile.onnx` — модель PaddleOCR, конвертированная и опубликованная RapidAI/RapidOCR.
Лицензия: Apache-2.0; тексты лицензий обоих проектов приложены в этом каталоге.
Авторство исходной модели: PaddlePaddle Authors. Распространение ONNX: RapidAI.

- Источник: https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv4/rec/cyrillic_PP-OCRv3_rec_mobile.onnx
- Реестр: https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr/default_models.yaml
- SHA-256: `1efb65bdc460af1c0e8733d005b20952b17ca5aac10ddb56c968333791c5eaa3`
- Препроцессинг PP-OCRv3: BGR, высота 48, ширина не менее 320 с сохранением пропорций и padding, нормализация в [-1,1].
- Описание входа PP-OCRv3: https://github.com/PaddlePaddle/PaddleOCR/blob/main/tools/infer/predict_rec.py
- Словарь встроен в ONNX metadata `character`; CTC blank добавляется первым, space последним.

Модель включается в пакет приложения. Скачивание во время обработки видео отсутствует.
