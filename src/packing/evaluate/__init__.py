import importlib
import pathlib


def import_all_tasks():
    base_path = pathlib.Path(__file__).parent
    # Edited to import only uncertainty quantification tasks, to avoid installing libraries for other tasks
    for pattern in ["task_uncertainty_quantification.py", "task_uncertainty_quantification_multi.py"]:
        for path in base_path.rglob(pattern):
            # Convert to dotted module path
            rel_path = path.relative_to(base_path.parent).with_suffix("")
            module = ".".join(['packing'] + list(rel_path.parts))
            importlib.import_module(module)
