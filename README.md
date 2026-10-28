# Instamart Customer Support Copilot

This project contains a synthetic Instamart customer-support dataset, a deterministic mock LLM, and a set of policy-driven evaluation scenarios for testing support workflows.

## Project contents

- `generate_data.py` creates synthetic customer, order, ticket, and policy data under `data/`.
- `mock_llm.py` acts as a local offline LLM for classification, policy lookup, and agent-style refund flows.
- `RunCopilot.py` provides a quick bootstrap entry point.

## Quick start

1. Create a virtual environment:
   `python -m venv .venv`
2. Activate it:
   - Windows PowerShell: `.\.venv\Scripts\Activate.ps1`
   - Command Prompt: `.venv\Scripts\activate.bat`
3. Install dependencies:
   `python -m pip install -r requirements.txt`
4. Generate the dataset:
   `python generate_data.py`
5. Run the project scripts as needed:
   `python RunCopilot.py`

## Notes

- The generated data is synthetic and deterministic.
- The repo includes a compatibility file named `requirements,txt.txt`, but `requirements.txt` is the standard file used by the setup steps above.
