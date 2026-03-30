import sys
from unittest.mock import MagicMock
sys.modules['pandas'] = MagicMock()
sys.modules['tqdm'] = MagicMock()
