"""Pin trellis_quant to its default reset width for the unit tests.

trellis_quant reads FLASHRUN_TRELLIS_WIDTH once at import time, and several
experiment drivers set it to 32 at import. Importing trellis_quant here first
keeps test results independent of collection order.
"""
import os

os.environ.pop("FLASHRUN_TRELLIS_WIDTH", None)
import trellis_quant  # noqa: E402,F401
