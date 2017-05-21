from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from get_time_features.holiday import str2datetime
from statsmodels.tsa.arima_model import ARIMA
from get_time_features.get_time_features import get_time_feature_vector
from main import *
import matplotlib.pyplot as plt
from intervals import *

def get_clusters(trainX):
    coefs = np.ones(n_days)
    znam = np.arange(n_days + 1, 1, -1)
    #print(znam.shape, coefs.shape)
    coefs /= znam
    coefs = np.array([coefs]).T
    def rescale_features(X):
        X[:n_days, :] *= coefs ** 2
        X[n_days:2*n_days, :] *= coefs ** 2
        X[2*n_days:3*n_days, :] *= coefs ** 2
        return X
    sc_list = []
    scl = StandardScaler()
    X = scl.fit_transform(trainX)
    X = rescale_features(X)
    cl = KMeans(n_clusters=12).fit(X)
    labels = cl.predict(X)
    centr = get_centriod(cl)
    return labels, centr


if __name__ == "__main__":
    create_ds = Data('learning_set/')
    ts = create_ds.create_train_set(isFull=False)
    labels = get_clusters(ts[0])


#uid = ts[2]
#for l in np.unique(labels):
#    n_labels = np.sum(labels == l)
#    print(n_labels, l)
#    print('=' * 50)
#    for i in range(min(n_labels, 5)):
#        sel_uid = uid[labels == l]
#        gr = create_ds.book4[sel_uid[i]]
#        plt.plot(gr[2])
#        plt.show()

