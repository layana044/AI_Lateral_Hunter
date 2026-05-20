@echo off
setlocal
cd /d "%~dp0"

set "CODEX_PY=C:\Users\USER\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

if exist "%CODEX_PY%" (
    "%CODEX_PY%" app.py
    goto :eof
)

where py >nul 2>nul
if %errorlevel%==0 (
    py -3 app.py
    goto :eof
)

where python >nul 2>nul
if %errorlevel%==0 (
    python app.py
    goto :eof
)

echo Python was not found. Install Python 3 and run:
echo pip install -r requirements.txt
echo python app.py
pause
