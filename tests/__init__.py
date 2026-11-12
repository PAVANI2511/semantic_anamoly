"""
Test suite initialization.
Ensures fast offline test execution using cached model weights without network retries.
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("SEMANTIC_SENTINEL_OFFLINE", "1")

