from importlib import import_module
from types import ModuleType
from typing import Dict

_modules: Dict[str, str] = {
    "burgers": "a_pinn.pdes.burgers",
    "allen_cahn": "a_pinn.pdes.allen_cahn",
    "shallow_water": "a_pinn.pdes.shallow_water",
}

def load(name: str) -> ModuleType:
    key = name.lower()
    if key not in _modules:
        raise KeyError(f"Unknown PDE '{name}'. Choose from {list(_modules)}")
    return import_module(_modules[key])


from . import burgers, allen_cahn, shallow_water

__all__ = [
    "burgers",
    "allen_cahn",
    "shallow_water",
    "load",
]