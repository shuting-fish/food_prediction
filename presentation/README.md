# Food Prediction Streamlit Presentation

Status: **Non-final**

This directory contains an evidence-bound, read-only presentation of the Food Prediction project.
It is based on `shuting-fish/master` commit `24c5757` and the active project-governance snapshot
used for the implementation.

## Scope

The presentation:

- explains the forecast target and canonical data concepts;
- separates candidate external enrichments from canonical raw data;
- maps the repository workflow at artifact level;
- makes open QA and evidence boundaries visible;
- does not load project data or model artifacts;
- does not run training, inference, comparison, SHAP, or feature-importance code;
- does not provide upload or download endpoints.

## Local run

From the repository root:

```powershell
py -m venv .venv-streamlit
& ".\.venv-streamlit\Scripts\python.exe" -m pip install -r ".\presentation\requirements.txt"
& ".\.venv-streamlit\Scripts\python.exe" -m streamlit run ".\presentation\streamlit_app.py"
```

## Local smoke test

```powershell
& ".\.venv-streamlit\Scripts\python.exe" -m unittest discover -s ".\tests" -p "test_streamlit_presentation.py"
```

Deployment is intentionally outside this implementation slice.
