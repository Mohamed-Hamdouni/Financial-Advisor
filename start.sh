#!/bin/bash

# Make the script executable if not already
chmod +x start.sh

# Ensure correct permissions on directories
chmod -R 755 vector_db
chmod -R 755 data

# Clear existing log file
echo "" > terminal.log

# Activate virtual environment and run streamlit
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment."
    exit 1
fi

echo "Starting Financial Advisor at $(date)" | tee -a terminal.log

# Run streamlit and tee output to terminal and log file
streamlit run app/main.py 2>&1 | tee -a terminal.log