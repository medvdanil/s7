import xgboost as xgb
from main import Data
import numpy as np
import pandas as pd


def predict_qlty(dt):
    (X_train, Y_train, uid_train) = dt.create_train_set()
    (X_test, uid_test) = dt.create_test_set()
    regr = xgb.XGBRegressor(colsample_bytree=1, subsample=0.9, min_child_weight=5, max_depth=6,
                            colsample_bylevel=0.8, learning_rate=0.1, n_estimators=200)
    regr.fit(X_train[Y_train >= 0], Y_train[Y_train >= 0])
    #print(X_test)
    preds = regr.predict(X_test)
    #print(preds)
    preds_int = np.round(preds)
    #print(preds_int)
    df = pd.DataFrame({'UID': uid_test, 'score': preds_int})
    #print(df)
    df.to_csv('marked_data_test.csv', sep=';')    


create_ds = Data('learning_set/')
predict_qlty(create_ds)
