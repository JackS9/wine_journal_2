#!/bin/bash
# setup.sh - Automation script for AI Wine Journal 2.0 environment setup

set -e

echo "🍷 Setting up AI Wine Journal 2.0 Environment..."

# 1. Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is not installed or not in PATH."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "✅ Detected Python version: $PYTHON_VERSION"

# 2. Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment ('venv')..."
    python3 -m venv venv
else
    echo "ℹ️  Virtual environment 'venv' already exists."
fi

# 3. Activate venv
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# 4. Upgrade pip & install requirements
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

if [ -f "requirements.txt" ]; then
    echo "📥 Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "⚠️  requirements.txt not found! Skipping package installation."
fi

# 5. Create required runtime directories
echo "📁 Ensuring required directories exist ('uploads', 'static')..."
mkdir -p uploads static

# 6. Check for .env file
if [ ! -f ".env" ]; then
    echo "⚙️  Creating template .env file..."
    cat << 'EOF' > .env
# AI Wine Journal Configuration
GEMINI_API_KEY=your_actual_gemini_api_key_here
EOF
    echo "⚠️  Created .env template. Please edit .env and replace with your GEMINI_API_KEY!"
else
    echo "✅ Existing .env file found."
fi

# 7. Ensure .gitignore exists
if [ ! -f ".gitignore" ]; then
    echo "📝 Creating default .gitignore..."
    cat << 'EOF' > .gitignore
.env
venv/
uploads/
*.db
__pycache__/
.DS_Store
EOF
    echo "✅ .gitignore created."
fi

echo ""
echo "🎉 Setup finished successfully!"
echo "-------------------------------------------------------"
echo "To run your app on your MacBook Pro:"
echo "  1. Make sure your GEMINI_API_KEY is in .env"
echo "  2. Activate the virtual environment:"
echo "       source venv/bin/activate"
echo "  3. Launch the FastAPI app:"
echo "       uvicorn main:app --reload --port 8000"
echo "-------------------------------------------------------"
