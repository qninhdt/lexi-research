"""Sentence-grader distillation research package."""

import sys

# Google Colab pre-installs an outdated, incompatible torchao 0.10.0 (< 0.16.0).
# Masking it in sys.modules ensures peft treats it as uninstalled rather than crashing.
if "torchao" not in sys.modules:
    try:
        import importlib.util

        if importlib.util.find_spec("torchao") is not None:
            import torchao
            import packaging.version

            if packaging.version.parse(
                getattr(torchao, "__version__", "0.0.0")
            ) < packaging.version.parse("0.16.0"):
                sys.modules["torchao"] = None
    except Exception:
        sys.modules["torchao"] = None
