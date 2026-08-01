import ast
import builtins
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "bank_churn_model.py"


def load_download_fallback():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "download_fallback"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {}
    exec(compile(module, MODULE_PATH, "exec"), namespace)
    return namespace["download_fallback"]


class DependencyLoadingTests(unittest.TestCase):
    def test_missing_kagglehub_reports_install_instructions_without_self_installing(self):
        download_fallback = load_download_fallback()
        original_import = builtins.__import__

        def import_without_kagglehub(name, *args, **kwargs):
            if name == "kagglehub":
                raise ImportError("not installed")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = import_without_kagglehub
        try:
            with self.assertRaisesRegex(RuntimeError, "pip install -r requirements.txt"):
                download_fallback()
        finally:
            builtins.__import__ = original_import


if __name__ == "__main__":
    unittest.main()
