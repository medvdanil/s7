from main import Data
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

def conf_intervals(graphic, labels, quantile):
    u_labels = np.unique(labels)
    interval_low = np.zeros((graphic.shape[1], u_labels.shape[0]))
    interval_high = np.zeros((graphic.shape[1], u_labels.shape[0]))
    
    #print(interval_low.shape)
    for l in u_labels:
        w = labels == l
        #sel_uid = uid[labels == l]
        #label_gr = np.zeros((sel_uid, graphic.shape[1]))
        label_gr = graphic[w, :]
        #print(label_gr.shape)
        gr = np.sort(label_gr, axis=0)
        
        qntil_low = int(label_gr.shape[0] * quantile)
        qntil_high = int(label_gr.shape[0] * (1 - quantile))
        #print(gr.shape)
        interval_low[:, l] = gr[qntil_low, : ]
        interval_high[:, l] = gr[qntil_high, : ]
    return interval_low, interval_high


def acum_all_gr(ds, gr_ind, correct_uids):
    gr = np.zeros((len(correct_uids), ds.book4[2][gr_ind].shape[0]))
    j = 0
    
    for i in np.unique(correct_uids):
        gr[j, :] = ds.book4[i][gr_ind]
        j += 1
    return gr

def get_centriod(clst, X, n_day=334):
    cntr = clst.cluster_centers_
    booking = np.zeros((cntr.shape[0], n_day))
    DC = np.zeros((cntr.shape[0], n_day))
    RASK = np.zeros((cntr.shape[0], n_day))
    
    for l in np.unique(clst.labels_):
        w = l == clst.labels_
        booking[l, :] = np.sum(X[w, :n_day], axis=0)/np.sum(w)
        DC[l, :] = np.sum(X[w, n_day:n_day * 2], axis=0)/np.sum(w)
        RASK[l, :] = np.sum(X[w, n_day * 2:n_day * 3], axis=0)/np.sum(w)


    #booking = cntr[:, :n_day]
    #DC = cntr[:, n_day:2*n_day]
    #RASK = cntr[:, 2*n_day:3*n_day]
    return booking, DC, RASK
    
n_days = 334
coefs = np.ones(n_days)
znam = np.arange(n_days + 1, 1, -1)
#print(znam.shape, coefs.shape)
coefs /= znam
coefs = np.array([coefs]).T
coefs *= coefs
def rescale_features(X):
    X[:n_days, :] *= coefs
    X[n_days:2*n_days, :] *= coefs
    X[2*n_days:3*n_days, :] *= coefs
    return X


if __name__ == "__main__":
    create_ds = Data('learning_set/')
    ts = create_ds.create_train_set(isFull=False)
    scl = StandardScaler()
    X = scl.fit_transform(ts[0])
    X = rescale_features(X)
    cl = KMeans(n_clusters=12).fit(X)
    labels = cl.predict(X)
    centr = get_centriod(cl, ts[0])

    book_data = acum_all_gr(create_ds, 1, ts[2])
    interv = conf_intervals(book_data, labels, 0.05)


