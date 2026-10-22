from importlib import import_module
from pathlib import Path

from .config import METHOD_NAMES
from .io import sha256


SOURCE_HASHES = {
    "found_formulas": "ac1ed0b7fa2884c9f326a861f68777db75fba7b18bcd8ef565b78fa225dabebc",
    "estimator_base": "67f791cf66e84968e318aa6b113f043eba268785dc198f8d9c4aaa023d8c6dd5",
}


def load_methods(context=None):
    modules = {
        name: import_module(f"src.packing.evaluate.uncertainty_quantification.{name}")
        for name in SOURCE_HASHES
    }
    actual = {name: sha256(Path(module.__file__)) for name, module in modules.items()}
    if actual != SOURCE_HASHES:
        raise ValueError("Repository formulas differ from the recorded experiment")
    context = context or {"supervised": False, "meta": {}}
    return [(name, getattr(modules["found_formulas"], name)(context=context)) for name in METHOD_NAMES]
