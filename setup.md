### How to Run Your Application on Your MacBook Pro

1. **Install Dependencies in your Terminal / Antigravity IDE**:
   ```bash
   pip install fastapi uvicorn google-genai requests beautifulsoup4 pydantic pillow sqlalchemy
   ```

2. **Set your Gemini API Key**:
   ```bash
   export GEMINI_API_KEY="your-gemini-pay-as-you-go-api-key"
   ```

3. **Start the Web Server**:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Open in Your Browser**:
   Navigate to `http://localhost:8000` to start scanning labels, running agent deep-dives, logging tasting memories, and chatting with your AI sommelier!

### Setting up API key

**using a `.env` file is the recommended best practice** for managing secret keys in Python projects. It keeps your API keys out of your source code and prevents accidentally committing them to version control.

Here is how to set it up:

---

### 1. `.env` File Format

Create a file named `.env` (with the leading dot, no file name before it) in your project root directory (the same folder where `main.py` lives):

```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

*Note: Do not put spaces around the `=` sign, and quotes are optional unless your key contains special characters or spaces.*

---

### 2. Install `python-dotenv`

Install the standard Python library for reading `.env` files:

```bash
pip install python-dotenv
```

---

### 3. Import and Load in Python

At the **very top** of your entry script (e.g., at the top of `main.py`, before importing any modules that rely on `GEMINI_API_KEY`), call `load_dotenv()`:

```python
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Now os.environ.get("GEMINI_API_KEY") will automatically read from .env
```

Since our modules (`research_agent.py` and `vivino_pipeline.py`) use `os.environ.get("GEMINI_API_KEY")`, calling `load_dotenv()` at the start of `main.py` makes the key automatically available across the entire app without changing the rest of your code!

---

### 💡 Pro Tip: Git Security
If you use Git or push your code to GitHub, create or open your `.gitignore` file and add `.env` to it:

```gitignore
.env
uploads/
*.db
```

This ensures your secret key and local database aren't accidentally published.



### Step-by-Step Setup Instructions for virtual environment

#### **1. Navigate to your project directory**
Open Terminal (or the integrated terminal in Antigravity IDE) and `cd` into your wine journal folder:
```bash
cd /Users/jacksmith/wine_journal_2
```

#### **2. Create the virtual environment**
Run Python's built-in `venv` module to create an isolated environment named `venv`:
```bash
python3 -m venv venv
```

#### **3. Activate the virtual environment**
Activate the environment so your terminal uses the virtual environment's Python and `pip`:
```bash
source venv/bin/activate
```
*(You will see `(venv)` prepended to your terminal prompt line, indicating it is active.)*

#### **4. Upgrade `pip` (Recommended)**
Ensure you have the latest package installer version:
```bash
pip install --upgrade pip
```

#### **5. Install dependencies from `requirements.txt`**
Install all required libraries at once:
```bash
pip install -r requirements.txt
```

---

### Working with the Virtual Environment

* **Starting your server**: Always make sure `(venv)` is active before running `uvicorn main:app --reload`.
* **Deactivating**: When you're done working, turn off the environment by typing:
  ```bash
  deactivate
  ```
* **Git `.gitignore`**: Make sure `venv/` is added to your `.gitignore` file so the environment directory isn't tracked in version control:
  ```gitignore
  .env
  venv/
  uploads/
  *.db
  ```