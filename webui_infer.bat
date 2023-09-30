@echo off

echo Running webui_infer.py...
venv\Scripts\python webui_infer.py

if errorlevel 1 (
    echo Error: Failed to run webui_infer.py.
    pause
    exit /b
)

pause
