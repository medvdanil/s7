import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from get_time_features.holiday import str2datetime
from statsmodels.tsa.arima_model import ARIMA
from get_time_features.get_time_features import get_time_feature_vector
from main import *


create_ds = Data('learning_set/')
ts = create_ds.create_train_set(isFull=False)

parameters = {'learning_rate':[0.05, 0.1, 0.25], 'n_estimators':[200], 'subsample':[0.9, 1], 'max_depth':[4, 6, 8], 
              'min_child_weight':[1, 2, 3, 5], 'colsample_bytree': [0.8, 1], 'colsample_bylevel':[0.8, 1]}
regr = xgb.XGBRegressor()
cv = GridSearchCV(regr, parameters, verbose=10, scoring='neg_median_absolute_error', n_jobs=1)
cv.fit(ts[0], ts[1])
