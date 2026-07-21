# Contributing

Thanks for improving the bank customer churn project.

## Development setup

1. Fork or clone the repository.
2. Create a focused branch from `main`.
3. Create and activate a Python virtual environment.
4. Install dependencies with `python -m pip install -r requirements.txt`.

## Quality checks

At minimum, verify the source compiles:

```bash
python -m py_compile bank_churn_model.py
```

Add automated tests when extracting reusable preprocessing, training, or evaluation functions. Use synthetic data in tests and documentation.

## Pull requests

Describe the data assumptions, modeling impact, and validation. Never commit customer records, credentials, or large generated artifacts.
