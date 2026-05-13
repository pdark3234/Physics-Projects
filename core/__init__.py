# Modified Gravity Studio - Core Package

import builtins
import sys

# Mock IPython display for non-Jupyter environments
# pytearcat calls display() which only exists in Jupyter
if not hasattr(builtins, 'display'):
    builtins.display = lambda *args, **kwargs: None

# Mock IPython modules if not present
if 'IPython' not in sys.modules:
    class MockDisplay:
        @staticmethod
        def display(*args, **kwargs):
            pass
        class Math:
            def __init__(self, *args, **kwargs):
                pass
        class Latex:
            def __init__(self, *args, **kwargs):
                pass
        # Mock IProgress widget
        class IProgress:
            pass
    class MockIPython:
        display = MockDisplay()
    sys.modules['IPython'] = MockIPython()
    sys.modules['IPython.display'] = MockDisplay()
    sys.modules['ipywidgets'] = MockDisplay()

# Force tqdm to use plain text mode (not notebook widgets)
import tqdm
if hasattr(tqdm, '_instances'):
    tqdm._instances.clear()
# Replace tqdm_notebook with regular tqdm
tqdm.tqdm_notebook = tqdm.tqdm
sys.modules['tqdm.notebook'] = tqdm
