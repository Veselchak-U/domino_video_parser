@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul
set "DOMINO_PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%~dp0.venv-gpu\domino-ready" if exist "%~dp0.venv-gpu\Scripts\python.exe" set "DOMINO_PYTHON=%~dp0.venv-gpu\Scripts\python.exe"
if not exist "%DOMINO_PYTHON%" (
    echo Ошибка: не найдена среда Python в .venv в корне проекта. 1>&2
    echo Выполните установку по инструкции README.md. 1>&2
    exit /b 1
)
"%DOMINO_PYTHON%" -m domino_video %*  --fish-variant withoutEggs --score-limit 50
exit /b %errorlevel%
