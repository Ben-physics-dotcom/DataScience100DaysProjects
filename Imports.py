import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
import pickle

import os
import sys
import shutil
import re

# Windows can't shell out to detect physical core count (WinError 2), so joblib/loky
# falls back with a warning; pin it to logical core count upfront to silence it.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))

# Kaggle
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

# Machine Learning Libraries
from scipy.stats import linregress

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


# own defined functions
from regression_functions import linreg, results_func

from saving_functions import saving_results, loading_results

sns.set_theme()
sns.set_context("paper")
