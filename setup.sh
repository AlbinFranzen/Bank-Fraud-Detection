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

# Step 3: Install `pip` if missing
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
source bank/bin/activate

# Step 6: Upgrade pip and install dependencies
echo "Upgrading pip and installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt || echo "No requirements.txt found, skipping."

# Step 7: Use `gofile-dl` to download the dataset
echo "Downloading dataset from Gofile..."
gofile-dl "https://gofile.io/d/70APEq" --output-dir Base/

echo "Setup complete!"
echo "Virtual environment is still active. Run 'deactivate' to exit."
echo "To reactivate the virtual environment later, run:"
echo "source env/bin/activate"