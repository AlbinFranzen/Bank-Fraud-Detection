@echo off

REM Setting up the project...
echo Setting up the project...

REM Step 1: Check for Python 3
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo Python3 is not installed. Please install it and try again.
    exit /b 1
)

REM Step 2: Install pip if not available
python -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing pip...
    python -m ensurepip
    if %errorlevel% neq 0 (
        echo Failed to install pip. Please check your Python installation.
        exit /b 1
    )
)

REM Step 3: Create the virtual environment if it doesn't exist
if not exist "bank" (
    echo Creating virtual environment...
    python -m venv bank
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment. Exiting.
        exit /b 1
    )
) else (
    echo Virtual environment already exists.
)

REM Step 4: Activate the virtual environment
echo Activating virtual environment...
call "bank\Scripts\activate.bat" || (
    echo Failed to activate virtual environment. Exiting.
    exit /b 1
)

REM Step 5: Upgrade pip and install dependencies
echo Upgrading pip and installing dependencies...
python -m pip install --upgrade pip
if exist "requirements.txt" (
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo Failed to install dependencies. Exiting.
        exit /b 1
    )
) else (
    echo No requirements.txt found, skipping dependency installation.
)

REM Step 6: Download the dataset with curl or wget
echo Downloading dataset from Google Drive...
if exist Data\ (
    echo Data folder exists.
) else (
    mkdir Data
)

curl -L -o Data\Base.csv "https://drive.usercontent.google.com/download?id=1gDHsL7iJsXjvkIY1VLb5IwlJ6AD236ic&export=download&confirm=t" || (
    echo Failed to download dataset. Please check the link.
    exit /b 1
)

REM Final message
echo Setup complete!
echo To activate the virtual environment, run:
echo bank\Scripts\activate.bat
echo To deactivate the virtual environment later, type "deactivate".
