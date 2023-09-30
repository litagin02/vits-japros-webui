@echo off
echo Updating the repository...

cd /d %~dp0

git pull

venv\Scripts\pip install -r requirements.txt

echo Update complete.
pause
