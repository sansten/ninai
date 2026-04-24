import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

nb_path = r"d:\Sansten\Projects\Ninai2\repos\ninai\notebooks\locomo_benchmark.ipynb"
nb_dir = os.path.dirname(nb_path)

with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=-1, kernel_name="python3")
print("Executing notebook:", nb_path)
ep.preprocess(nb, {"metadata": {"path": nb_dir}})

with open(nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Notebook execution completed and saved.")
