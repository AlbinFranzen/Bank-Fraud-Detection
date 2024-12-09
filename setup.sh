#!/bin/bash

echo "Setting up the project..."

# Step 1: Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install it and try again."
    exit 1
fi

# Step 2: Install `python3-venv` if missing
if ! python3 -m venv --help &> /dev/null; then
    echo "Installing python3-venv..."
    sudo apt update
    sudo apt install -y python3-venv
fi

# Step 3: Ensure pip is installed
if ! command -v pip &> /dev/null; then
    echo "Installing python3-pip..."
    sudo apt install -y python3-pip
fi

# Step 4: Create the virtual environment if it doesn't exist
if [ ! -d "bank" ]; then
    echo "Creating virtual environment..."
    python3 -m venv bank
else
    echo "Virtual environment already exists."
fi

# Step 5: Activate the virtual environment
echo "Activating virtual environment..."
source bank/bin/activate || { echo "Failed to activate virtual environment. Exiting."; exit 1; }

# Step 6: Ensure pip is upgraded and dependencies are installed
echo "Upgrading pip and installing dependencies..."
python3 -m pip install --upgrade pip
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "No requirements.txt found, skipping dependency installation."
fi

# Step 7: Install `gofile-dl` if not installed
if ! command -v gofile-dl &> /dev/null; then
    echo "Installing gofile-dl..."
    pip install gofile-dl
fi

# Step 8: Use `gofile-dl` to download the dataset
echo "Downloading dataset from Gofile..."
gofile-dl "https://gofile.io/d/70APEq" --output-dir Base/ || { echo "Failed to download dataset. Please check the link."; exit 1; }

# Step 9: Final message
echo "Setup complete!"
echo "Virtual environment is still active. Run 'deactivate' to exit."
echo "To reactivate the virtual environment later, run:"
echo "source bank/bin/activate"
