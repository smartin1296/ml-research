# Symlink to shared device utilities
import sys
from pathlib import Path

# Add the shared core module to path
sys.path.append(str(Path(__file__).parents[2] / 'rnn' / 'core'))

from device_utils import *