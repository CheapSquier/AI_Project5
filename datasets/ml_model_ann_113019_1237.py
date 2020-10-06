import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import math
import os

def normalizeInputs(inputCol):
    print("*** In normalizeInput, not confident we're correctly normalizing. ***")
    # Should we be modifying in place or returning a copy?
    return normalize(inputCol)

def partitionData(df):
    #Maybe just do this in ml_main
    return trainDF, validDF