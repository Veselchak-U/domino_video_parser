@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo Ошибка: не найдена среда Python в .venv в корне проекта. 1>&2
    echo Выполните установку по инструкции README.md. 1>&2
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -m domino_video %*  --fish-variant withoutEggs --score-limit 50
exit /b %errorlevel%
