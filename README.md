## Create an environment

```
conda create -n lab3 python=3.11 -y
conda activate lab3

```

## Install Python packages

```
pip install --upgrade setuptools wheel pyquery
pip install -r requirements.txt

```

**Troubleshooting**:
- If `pip` is not found after creating the environment, run: `python -m ensurepip --upgrade`
- If you get `ModuleNotFoundError: No module named 'pkg_resources'`, run: `pip install setuptools==68.2.2`

## Run the project
```
flask --app flaskr run --debug --port 5001
```

Then open http://127.0.0.1:5001 in your browser.


# Run evaluation and observe result
This is the Rationale for Algorithm Selection
```
python evaluate_algorithms.py
```
