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

runpy.run_path(str(ROOT / "src" / "vrpr" / "web" / "app.py"), run_name="__main__")
