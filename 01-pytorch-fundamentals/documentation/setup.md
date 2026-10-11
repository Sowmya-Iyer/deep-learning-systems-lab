## Setup

Commands are shown for Bash. Where another shell differs, the alternative is listed.

### 1. Clone the repository and enter this folder

```
git clone <repository-url>
cd deep-learning-systems-lab/01-pytorch-fundamentals
```

Clone into a folder that is not synced by OneDrive or another cloud-sync tool. The virtual environment is several GB and contains thousands of files.

### 2. Create the virtual environment

```
python -m venv .torch-venv
```

The folder name starts with a dot, so `ls` in Git Bash does not show it. Use `ls -a` to confirm it was created.

### 3. Activate it

| Shell | Command |
|---|---|
| Windows Git Bash | `source .torch-venv/Scripts/activate` |
| Windows PowerShell | `.torch-venv\Scripts\Activate.ps1` |
| Windows Command Prompt | `.torch-venv\Scripts\activate.bat` |
| Linux | `source .torch-venv/bin/activate` |

The prompt should now show `(.torch-venv)`.

### 4. Install the dependencies

With an NVIDIA GPU:

```
python -m pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu132
```

The `--extra-index-url` option is required. `requirements.txt` pins CUDA builds of PyTorch (versions ending in `+cu132`), which are hosted on PyTorch's own package index and not on PyPI. The download is several GB.

Without an NVIDIA GPU, install the CPU-only build instead of using `requirements.txt`:

```
python -m pip install --upgrade pip
pip install torch torchvision
```

### 5. Verify

```
python startup.py
```

Expected output on a GPU machine: `True`, followed by the name of the GPU. On a CPU-only install the first line is `False`.

## Everyday use

Activate the environment (step 3) each time you open a new terminal in this folder, then run scripts with `python`:

```
python your_script.py
```

Leave the environment with:

```
deactivate
```

## Adding or updating a package

With the environment active, install the package and then re-save the pinned list so others get the same versions.

Git Bash or Linux:

```
pip install <package>
pip freeze > requirements.txt
```

PowerShell:

```
pip install <package>
pip freeze | Set-Content requirements.txt
```

Commit the updated `requirements.txt`.

In Windows PowerShell, use the `Set-Content` form and not `>`, which writes the file as UTF-16 and makes Git treat it as binary.

## Rebuilding the environment

A virtual environment stores its own absolute path, so it breaks if the project folder is moved or renamed. It is safe to delete and recreate at any time.

Git Bash or Linux:

```
deactivate
rm -rf .torch-venv
```

PowerShell:

```
deactivate
Remove-Item -Recurse -Force .torch-venv
```

Then repeat setup steps 2 to 5.

## Notes

- **`.torch-venv/` must stay out of Git.** It is listed in `.gitignore`. Only `requirements.txt` is committed.
- **Changing the CUDA version.** To move to a different CUDA build, uninstall with `pip uninstall -y torch torchvision`, install with the command from <https://pytorch.org/get-started/locally/>, re-save `requirements.txt`, and update the index URL in step 4 of this file to match.

## Troubleshooting

**`AssertionError: Torch not compiled with CUDA enabled`.** The CPU-only build of PyTorch is installed. Replace it with the CUDA build:

```
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

**`startup.py` prints `False` with the CUDA build installed.** PyTorch cannot see the GPU. Run `nvidia-smi`; if it fails or reports a CUDA version below 13.2, update the NVIDIA driver.

**`No matching distribution found for torch==...+cu132`.** The `--extra-index-url` option was left off the install command in step 4, or the installed Python version is outside 3.10 to 3.14.

**`python -m venv` appears to do nothing (Windows).** Either the folder was created and is hidden from `ls` (check with `ls -a`), or `python` is the Microsoft Store placeholder. Run `python --version`; if it prints nothing, use `py -m venv .torch-venv` or install Python from <https://www.python.org/downloads/> with "Add python.exe to PATH" ticked.

**"Running scripts is disabled on this system" when activating in PowerShell.** Allow local scripts for your user once, then activate again:

```
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```
