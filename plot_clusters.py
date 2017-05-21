from clustering import *
from sys import argv
from os.path import isfile
from sklearn.externals import joblib
from main import *
from matplotlib.pyplot import cm

def rm_bad_data(x, i_n):
    if i_n != 0:
        return x
    for i in range(x.shape[0]):
        for j in range(1, x.shape[1] - 1):
            if x[i][j] * 0.3 > x[i][j-1] + x[i][j+1]:
                x[i][j] = (x[i][j-1] + x[i][j+1]) / 2
    return x
    
if __name__ == "__main__":
    if isfile('data.dump'):
        d = joblib.load('data.dump')
    else:
        d = Data('/files/data/learning_set/')
        joblib.dump(d, "data.dump", compress=9)
    x, y, uids = d.create_train_set(isFull=False)
    labels, centers = get_clusters(x)
    
    intv_min = []   
    intv_max = []
    names = ['booking', 'DC', "RASK"]
    for i_n in range(len(names)):
        imn, imx = conf_intervals(rm_bad_data(x[:, n_days * i_n:n_days * (i_n + 1)], i_n), labels, 0.05)
        intv_min.append(imn)
        intv_max.append(imx)
        
    def plt_k(k):
        for i, lbl in enumerate(labels):
            if lbl != k:
                continue
            plt.plot(d.book4[uids[i]][1])
        plt.show()
        
    colors = cm.rainbow(np.linspace(0, 1, 1+ labels.max()))
    plt.rc('font', family='DejaVu Sans')

    for i_n in range(len(names)):
        for k in range(labels.max() + 1):
            values = centers[i_n][k]
            e_min = values - intv_min[i_n][:, k]
            e_max = intv_max[i_n][:, k] - values
            plt.errorbar(np.arange(n_days), values, yerr=np.vstack((e_min, e_max)), ecolor=colors[k], color=colors[k])
            plt.plot(values, linewidth=3, color=colors[k], label="%d, профиль эволюции загрузки" % k)
        plt.legend(prop={'size':6})
        plt.savefig('IBP_graphics_%s.png' % names[i_n])
        plt.clf()
        
    for i_n in range(len(names)):
        bc = pd.DataFrame([[0] * 5], columns=['id', 'down', 'up', 'depth', 'booking'])
        cnt = 0
        for k in range(labels.max() + 1):
            for i in range(len(centers[i_n][k])):
                bc.loc[cnt] = [k, e_min[i], e_max[i], max_day - i, centers[i_n][k][i]]
                cnt += 1
        bc.to_csv(clusters_file % names[i_n], sep=';')
