"""`python -m lexi_research.cli` — the same entry point as the `lexi` script.

Kept because the Colab runtime clones the repo rather than installing the
package, so `[project.scripts]` never materialises there.
"""

from __future__ import annotations

from . import main

raise SystemExit(main())
