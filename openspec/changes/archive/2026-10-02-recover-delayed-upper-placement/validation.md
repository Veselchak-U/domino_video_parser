# Проверка восстановления задержанной выкладки

Базовая ревизия: `81d2a5daa94434954ce6b10d33e78ab37581b7cc` плюс локальные изменения vision, stone_recovery, нового тестового файла и двух фикстур.

## Причины и реальные доказательства

Исходник: `Record_2026-09-29-14-41-58_ef23194d097aa6e1b107d0f095c4226c.mp4`, SHA256 `9ae9c1017880af8d0891efdbf0efbe2626f3816fb1ec2f4ba3e3491d9993560c`.

- Начальный полный pipeline воспроизвёл ошибку кона 1, exit 1. Время 4-5 ошибочно совпадало с 2-6: 68.104922 с. На шести исходных кадрах 60.909056–60.992400 с читается полёт 4-5; bbox y=83–116 исключался обычной областью стола. Кадр 60.944256 с сохранён в `tests/fixtures/upper_flight.png`.
- После восстановления 4-5 обнаружился отказ кона 2: 1-1 ошибочно имел два места пропуска. Непрерывный native-трек с 216.981567 до 218.466222 с был отсечён по последней секунде перед раскрытием 218.582622 с. Сохранение целого трека связывает его с коротким событием 217.064267 с; следующий 1-6 остаётся отдельным ходом.
- Cached replay реальных наблюдений: 23/27 действий; первый кон — fish, 14:42; второй — emptyHand, 14:76. 4-5 сыгран местом 3 до 2-6 места 4; последние два камня второго кона — 1-1 места 2 и 1-6 места 3. Эти наблюдения сохранены в `tests/fixtures/delayed_upper_placement.json.gz`.

## Падающие регрессии

Все команды Python используют `.venv/Scripts/python.exe -X utf8`.

- `-m pytest -q tests/test_delayed_upper_placement.py`: до реализации 4 failed / 6 passed. Падали окно, перенос incoming/settled и верхняя полоса; негативные сценарии проходили.
- `-m pytest -q tests/test_delayed_upper_placement.py -k terminal_reading`: до исправления непрерывной связи 1 failed / 1 passed.
- `-m pytest -q tests/test_delayed_upper_placement.py -k player_indicator`: до сохранения ненадёжного индикатора 2 failed / 1 passed.

Логи: `.analysis/edge-ambiguity/red.log`, `red-terminal.log`, `red-unreliable.log`. Промежуточный полный набор до последнего исправления: 501 passed; он не подменяет итоговую проверку.

## Границы проверки

В рабочей .venv DirectML недоступен, поэтому реальные прогоны используют CPU OCR. Проверка с подменой не считается проверкой GPU. Полного вручную размеченного эталона новой партии нет: сверяются конкретные визуальные доказательства, контракт JSON, полный игровой replay и распознанное табло. Прежние результаты сравниваются целиком как разобранные JSON.

## Итоговые тесты

На итоговом состоянии кода и тестов, включая ненадёжный индикатор:

- `-m pytest -q tests/test_delayed_upper_placement.py tests/test_stone_recovery.py tests/test_terminal_placement.py tests/test_reconstruct.py`: 89 passed, exit 0.
- `-m pytest -q`: 504 passed in 62.64s, exit 0.
- `-m ruff check domino_video tests`: exit 0.
- `-m ruff format --check domino_video tests`: exit 0, 54 файла соответствуют формату.

Лог `.analysis/edge-ambiguity/full-checks.log`. После этого код и тесты не менялись; подготовка отчёта, архивирование и коммит не требуют нового pytest.

## Полный прогон видео

Команда: `.venv/Scripts/python.exe -X utf8 .analysis/edge-ambiguity/complete/run_complete.py`, exit 0. Runner копирует все четыре MP4 из `in` и `in/ready` в отдельный каталог `complete/ready`, вызывает CLI с `--fish-variant withoutEggs --score-limit 50 --workers 24 --device auto`, сохраняет SHA256 до/после и оставляет оригиналы на месте. Суммарное время 148.578 с; это контроль корректности, не сравнительный benchmark.

Проверка: `.venv/Scripts/python.exe -X utf8 .analysis/edge-ambiguity/complete/validate_results.py`, exit 0. Все четыре JSON прошли GameValidator, все четыре отчёта имеют status=ok.

| Запись | Результат |
|---|---|
| 2026-09-28 13:58 | Полное совпадение разобранного JSON с out/2026-09-28-13-58-source-001-game-001.json |
| 2026-09-28 20:45 | Полное совпадение разобранного JSON с out/2026-09-28-20-45-source-001-game-001.json |
| 2026-09-28 20:52 | Полное совпадение разобранного JSON с out/2026-09-28-20-52-source-001-game-001.json |
| 2026-09-29 14:41 | Два кона, 23/27 действий; fish 14:42, затем emptyHand 14:76, победитель Team A; OCR табло согласован с расчётом |

Новый результат: `.analysis/edge-ambiguity/complete/out/2026-09-29-14-41-source-004-game-001.json`.

Логи и воспроизводимость: `.analysis/edge-ambiguity/complete/{run.log,run.json,validation.log,validation.json,runtime.json}`. Все четыре SHA256 до/после совпали. Три прежних хеша: `09d33287088535cfd2c06170d7741a38e008a2e8b60dc91c4f06a15ae9fbe065`, `1893ec38255aefb16c9c80780be2d7948c3174f6d8598fc416511196ab270dfe`, `42ff9fd0e8a584acbca4cc22e40d6de9b1a6b2a82bd6df2745c39f680f6f998e`.

Среда: rapidocr-onnxruntime 1.4.4, onnxruntime 1.30.0, opencv-python 5.0.0.93, av 18.1.0. Локальная модель `cyrillic_PP-OCRv3_rec_mobile.onnx`, SHA256 `1efb65bdc460af1c0e8733d005b20952b17ca5aac10ddb56c968333791c5eaa3`. Во всех четырёх прогонах CPU decode (GPU медленнее на пробе) и реальный CPU OCR (DirectML недоступен).

Итоговый diff приложения и тестов просмотрен; `git diff --check` прошёл. Посторонних правок в рабочем дереве нет.

## Архивирование

Штатный скрипт tools/ai/archive_openspec_change.sh recover-delayed-upper-placement завершился с exit 0. Встроенная проверка до архива: 5 passed, 0 failed; после архива: 4 passed, 0 failed. Синхронизированы два ADDED требования video-game-export; изменение перемещено в archive/2026-10-02-recover-delayed-upper-placement. Другой активный change сохранён. Лог: .analysis/edge-ambiguity/archive.log.
