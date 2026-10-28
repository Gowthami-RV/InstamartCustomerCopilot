"""Bootstrap helper for the Instamart Customer Support Copilot.

This keeps the project runnable from a single entry point and makes the
workspace initialization explicit for a clean local setup.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQ_FILE = ROOT / "requirements.txt"
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"

print("Instamart Customer Support Copilot initialized.")
print(f"Project root: {ROOT}")
print(f"Requirements file: {REQ_FILE}")
if VENV_PYTHON.exists():
    print(f"Virtual environment: {VENV_PYTHON}")
print("Next steps:")
print("  1. Activate the virtual environment")
print("  2. python -m pip install -r requirements.txt")
print("  3. python generate_data.py")
