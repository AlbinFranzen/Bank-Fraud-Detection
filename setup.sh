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

# Step 7: Check for and download the dataset
DATASET_PATH="Data/Base.csv"
if [ -f "$DATASET_PATH" ]; then
    echo "Dataset already exists at $DATASET_PATH. Skipping download."
else
    echo "Dataset not found. Downloading dataset from Google Drive..."
    wget "https://drive.usercontent.google.com/download?id=1gDHsL7iJsXjvkIY1VLb5IwlJ6AD236ic&export=download&confirm=t" -O "$DATASET_PATH" || { 
        echo "Failed to download dataset. Please download Base.csv from https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022 and place it in $DATASET_PATH."; 
        exit 1; 
    }
fi

# Step 8: Open the app and navigate to the URL
echo "Opening /flask_gui/app.py..."
if [ -f "flask_gui/app.py" ]; then
    python flask_gui/app.py &
    FLASK_PID=$!
    echo "Waiting for the Flask app to start..."
    sleep 2  # Give Flask time to start
    echo "Navigating to http://127.0.0.1:5000 in your default browser..."
    if command -v xdg-open &> /dev/null; then
        xdg-open http://127.0.0.1:5000
    elif command -v open &> /dev/null; then
        open http://127.0.0.1:5000
    else
        echo "Could not detect a command to open the browser. Please open http://127.0.0.1:5000 manually."
    fi
    wait $FLASK_PID  # Wait for the Flask process to exit
else
    echo "The file /fraud/app.py does not exist. Skipping..."
fi

# Step 9: Final message
echo "Setup complete!"
echo "To activate the virtual environment, run:"
echo "source bank/bin/activate"
echo "To deactivate the virtual environment later, run 'deactivate' to exit."
