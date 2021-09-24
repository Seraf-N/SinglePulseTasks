""" Functions for spliting a baseband file to smaller files with less channels.
"""
import os
import sys
import time

import numpy as np
import astropy.units as u
from astropy.table import QTable

from baseband_tasks.functions import Square


class StreamSpliter:
    """A class for spliting the baseband stream.
    """
    def __init__(self, source_files):
        pass
