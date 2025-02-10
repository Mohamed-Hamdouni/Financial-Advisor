#!/bin/bash

echo "🚀 Setting up Financial Advisor environment..."

# Vérification de Python 3.10
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not found"
    exit 1
fi

# Création de l'environnement virtuel
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Mise à jour de pip
echo "🔄 Updating pip..."
pip install --upgrade pip

# Installation des dépendances principales
echo "📚 Installing main dependencies..."
PACKAGES=(
    "torch"
    "openai"
    "langchain"
    "langchain-community"
    "langchain-openai"
    "numpy"
    "pandas"
    "sentence_transformers"
    "chromadb"
    "pypdf"
    "matplotlib"
)

for package in "${PACKAGES[@]}"; do
    echo "Installing $package..."
    pip install $package
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install $package. Check the logs for errors."
        exit 1
    fi
done

# Installation des dépendances additionnelles
echo "📚 Installing additional dependencies..."
ADDITIONAL_PACKAGES=(
    "PyPDF2"
    "tiktoken"
    "feedparser"
    "yfinance"
    "dateparser"
    "umap-learn"
    "python-dotenv"
    "streamlit"
)

for package in "${ADDITIONAL_PACKAGES[@]}"; do
    echo "Installing $package..."
    pip install $package
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install $package. Check the logs for errors."
        exit 1
    fi
done
pip install -U sentence-transformers

# Création des dossiers nécessaires avec les bonnes permissions
echo "📁 Creating project directories..."
mkdir -p data vector_db
chmod 755 data vector_db

# Nettoyer la base de données existante si nécessaire
echo "🧹 Cleaning up existing database..."
rm -rf vector_db/*

# Vérification de l'installation
echo "✅ Verifying installation..."
python -c "import torch; import openai; import langchain; import streamlit" || echo "❌ Some packages failed to install properly"

echo "🎉 Setup complete! To activate the environment:"
echo "source venv/bin/activate"
