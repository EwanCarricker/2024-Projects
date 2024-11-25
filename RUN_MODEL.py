#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Imports

import pandas as pd
import scipy.stats as stats
from datetime import datetime, timedelta
from scipy.stats import binom
import numpy as np
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2
import statsmodels.api as sm
import importlib
from itertools import product
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

# Import from previous file
import future_fixtures
importlib.reload(future_fixtures)
from future_fixtures import df_final

import sys
import os
sys.stdout = open(os.devnull, 'w')


# In[23]:


df_final['Cover?'] = df_final['Home Cover %'].apply(lambda x: 'Yes' if x > 53 else ('Yes' if x < 47 else 'No'))

print(df_final[['Home Team', 'Away Team','Home Cover %', 'Away Cover %','Cover?']].sort_values(by=['Home Cover %']))


# In[ ]:




