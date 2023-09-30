@echo off

echo Creating virtual environment...
python -m venv venv

if errorlevel 1 (
    echo Error: Failed to create virtual environment.
    exit /b
)

echo Installing torch and torchaudio...
venv\Scripts\pip install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

if errorlevel 1 (
    echo Error: Failed to install torch and torchaudio.
    exit /b
)

echo Installing packages from requirements.txt...
venv\Scripts\pip install -r requirements.txt

if errorlevel 1 (
    echo Error: Failed to install packages.
    exit /b
)

echo Downloading pretrained model...
if not exist pretrained mkdir pretrained
curl -L "https://huggingface.co/litagin/vits-japros-pretrained/resolve/main/pretrained.pth" -o "pretrained\pretrained.pth"

if errorlevel 1 (
    echo Error: Failed to download pretrained model.
    exit /b
)

if not exist "weights\pretrained\" mkdir "weights\pretrained\"

if not exist "weights\pretrained\pretrained.pth" (
    echo Copying pretrained model to weights/pretrained/...
    copy "pretrained\pretrained.pth" "weights\pretrained\pretrained.pth"
)

echo Setup complete.

pause