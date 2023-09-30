@echo off

echo Running webui_train.py...
venv\Scripts\python webui_train.py

if errorlevel 1 (
    echo Error: Failed to run webui_train.py.
    pause
    exit /b
)

pause
