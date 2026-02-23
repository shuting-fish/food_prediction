# Readme

## Virtual Env

### Option 1: Using `uv`

Create virtualenv using `uv`

```
uv venv --python <python-version> --seed
```

Activate venv with

```
source venv/bin/activate
```

### Option 2: Using python directly

```
python3 -m venv "env_name"
```

## Install Dependencies

Install dependencies inside the venv with the following command:

```
pip install -r requirements.txt
```

```
python --version
```

# Branching policy (team):

- master is the only long-lived branch and is treated as stable.
- No direct commits to master.
- Each change is developed on a feature branch: feature/<name>-<topic>
- Push feature branches to your fork (origin) and open a PR to upstream/master.
- After merge: delete the feature branch (remote + local).
- Keep your fork's master in sync with upstream/master regularly.

## Standard workflow (PowerShell)

```powershell
git checkout master
git fetch upstream
git merge upstream/master
git push origin master

git checkout -b feature/<name>-<topic>
git push -u origin feature/<name>-<topic>
