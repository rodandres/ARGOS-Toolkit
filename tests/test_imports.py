import importlib
import pkgutil

import argos


def test_all_argos_modules_import():
    for module in pkgutil.walk_packages(
        argos.__path__,
        argos.__name__ + ".",
    ):
        importlib.import_module(module.name)