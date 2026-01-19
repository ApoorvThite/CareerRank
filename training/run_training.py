"""
Wrapper script to fix packaging import issue and run training.
"""

import sys
import os

# Fix packaging import issue
try:
    import packaging
except:
    pass

# Add local site-packages to path
sys.path.insert(0, '/Users/apoorvthite/.local/lib/python3.12/site-packages')

# Now import and run the training script
import importlib.util
spec = importlib.util.spec_from_file_location("train_bi_encoder", "training/train_bi_encoder.py")
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)
