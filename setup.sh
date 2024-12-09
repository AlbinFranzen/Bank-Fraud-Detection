#!/bin/bash

echo "Setting up the project..."

# Step 1: Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install it and try again."
    exit 1
fi

# Step 2: Create a virtual environment
echo "Creating virtual environment..."
python3 -m venv bank

# Step 3: Activate the virtual environment
echo "Activating virtual environment..."
source bank/bin/activate

# Step 4: Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Step 5: Use `gofile-dl` to download the dataset
echo "Downloading dataset from Gofile..."
gofile-dl "https://gofile.io/d/70APEq" --output-dir Base/

echo "Setup complete!"
echo "Virtual environment is still active. Run 'deactivate' to exit."
echo "To reactivate the virtual environment later, run:"
echo "source env/bin/activate"