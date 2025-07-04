# 🚀 Snellius Complete Setup Guide (Genoa Node with `uv`)

This comprehensive guide helps you set up a Python environment on the Snellius supercomputer using [`uv`](https://github.com/astral-sh/uv), a fast package manager and virtual environment tool.

---
## 🚦 Step 0: Launch an Interactive Session (Important)

First, start an interactive session on a Genoa node:

```bash
srun --partition=genoa --ntasks=1 --cpus-per-task=4 --mem=8G --time=00:30:00 --pty bash
```

## 🖥️ Step 1: Load Required Modules

Clear old modules and load Python 3.11:

```bash
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
```

---

## 📦 Step 2: Install `uv`

Install `uv` locally with `pip`:

```bash
pip install --user uv
```

Make sure your local binary folder is in your `PATH`:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

To permanently add this to your path:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
```

---

## 🐍 Step 3: Create & Activate Virtual Environment

Use `uv` to set up your virtual environment:

```bash
uv venv .venv
source .venv/bin/activate
```

Deactivate when finished:

```bash
deactivate
```

---

## 📜 Step 4: Install Project Dependencies

- **Using `requirements.txt`**:

```bash
uv pip install -r requirements.txt
```

- **Using `pyproject.toml` (PEP 621)**:

```bash
uv pip install .
```

- **Editable install (for development)**:

```bash
uv pip install -e .
```

---

## ✅ Step 5: Verify Installation

Check your environment is set correctly:

```bash
which python
python --version
uv --version
```

Python should point to `.venv/bin/python`.

---

## 🧠 Step 6: Confirm Genoa Node

Ensure you're running on a Genoa CPU node:

```bash
lscpu | grep "Model name"
```

Should output:

```
AMD EPYC 9654 96-Core Processor
```

Check available memory:

```bash
free -h
```

---

## 💻 Step 7: Running Interactive Jobs

To launch an interactive job on a Genoa node:

```bash
srun --partition=genoa --ntasks=1 --cpus-per-task=2 --mem=4G --time=00:10:00 --pty bash
```

Remember:
- Genoa nodes charge a minimum of 24 CPUs.
- Adjust `--cpus-per-task` and `--mem` according to your needs.

---

## 🔁 Step 8: Re-enter Environment (Future Sessions)

Quick commands to get back into your Python environment:

```bash
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source .venv/bin/activate
```

---

## 💡 Additional Tips

- Use `uv lock` to manage exact dependency versions for reproducibility:

```bash
uv pip freeze > requirements.txt
```

- Always activate the virtual environment at the start of your session.


## 📌 Step 9: Running Batch Jobs on Snellius

To submit longer-running or computationally intensive experiments, create a **Slurm batch job script** as follows:

**`run_experiments.slurm`:**
```bash
#!/bin/bash
#SBATCH --job-name=scb_experiments
#SBATCH --partition=genoa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=scb_experiments-%j.out
#SBATCH --error=scb_experiments-%j.err

# Load required modules
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

# Activate your uv-managed virtual environment
source .venv/bin/activate

# Run your experiments
python -m npsem.NIPS2018POMIS_exp.test_bandit_strategies
```

---

✅ You're ready to go! Happy coding!