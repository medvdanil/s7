import pandas as pd
from main import Data
from clustering import get_clusters


def get_descr(labels, uids):
    dscr = []
    for l in np.unique(labels):
        w = labels == l
        l_uids = uids[w]
        descr = "Profile for flight with uids: "
        for i in l_uids:
            descr += str(i) + ' '
        dscr.append(descr)
        
    df = pd.DataFrame({"id": labels, "description": dscr})
    
    return df
        
if __name__ == "__main__":
    create_ds = Data('learning_set/')
    ts = create_ds.create_train_set(isFull=False)
    labels = get_clusters(ts[0])
    dsr = get_descr(labels, ts[2])
    print(dsr)
