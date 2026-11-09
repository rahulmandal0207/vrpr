"""Serve the VRPR web page as a public demo, for example on Streamlit Community Cloud.

Hosting services install ``requirements.txt`` and run this file. On your own computer, use
``vrpr web`` instead, which serves the same page without the public-demo limits.
"""

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = str(ROOT / "src")

# The host installs the dependencies but not this package, so import it from the source tree.
if SOURCE not in sys.path:
    sys.path.insert(0, SOURCE)
os.chdir(ROOT)
os.environ["VRPR_PUBLIC_DEMO"] = "1"

from vrpr.cpu import THREAD_VARIABLES, available_cpus  # noqa: E402

# Hosts cap CPU time below the cores they report. Size thread pools to the cap before PyTorch
# loads, and let idle threads sleep instead of spinning through the allowance.
for variable in THREAD_VARIABLES:
    os.environ.setdefault(variable, str(available_cpus()))
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

runpy.run_path(str(ROOT / "src" / "vrpr" / "web" / "app.py"), run_name="__main__")
