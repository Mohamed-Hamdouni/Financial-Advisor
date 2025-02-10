#!/bin/bash

# Make the script executable if not already
chmod +x start.sh

# Ensure correct permissions on directories
chmod -R 755 vector_db
chmod -R 755 data

# Activate virtual environment and run streamlit
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment."
    exit 1
fi

echo "Starting Financial Advisor at $(date)"

# Run streamlit app
streamlit run app/main.py
