import os
import subprocess

scripts_dir = os.path.dirname(os.path.abspath(__file__))
projekt_root = os.path.abspath(os.path.join(scripts_dir, ".."))
RAW_DATA_DIR = os.path.join(projekt_root, "raw_data")

if not os.path.isdir(RAW_DATA_DIR):
    raise FileNotFoundError(f"Directory not found: {RAW_DATA_DIR}")

for subdir in os.listdir(RAW_DATA_DIR):
    subdir_path = os.path.join(RAW_DATA_DIR, subdir)
    clean_notebook_path = os.path.join(subdir_path, "clean_data.ipynb")

    if os.path.isdir(subdir_path) and os.path.isfile(clean_notebook_path):
        print(f"Executing: {clean_notebook_path}")

        try:
            subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", clean_notebook_path], check=True)
            print(f"Executed successfully: {clean_notebook_path}\n")
        except subprocess.CalledProcessError as e:
            print(f"Execution failed: {clean_notebook_path}, Error: {e}\n")

print("All `clean_data.ipynb` scripts executed.")
