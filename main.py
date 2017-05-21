from get_time_features.holiday import *
from os import path
import pandas as pd
import numpy as np
from get_time_features.get_time_features import *

depths_fname = 'Hackathon_DATA.csv'
rask_dc_fname = 'Hackathon_RASK_DC.csv'
test_uids_fname = 'Hackathon_test.csv'
labels_fname = 'marked_data.csv'
volumes_fname = 'Market_vol.csv'
clusters_file = 'flight_%s_graph_data.csv'

min_day = -3
max_day = 330
n_days = max_day - min_day + 1

def fix_str_int(a):
    return int(float(str(a).replace(',', '.')) + 0.5)
    
fix_str_int_arr = np.vectorize(fix_str_int)

def fix_str_float(a):
    return float(str(a).replace(',', '.'))
    
fix_str_float_arr = np.vectorize(fix_str_float)

class Data:
    def __init__(self, datadir):
        d = pd.read_csv(path.join(datadir, depths_fname), sep=';')
        d = d[d['GroupInfo'] == 'Depth']
        d = d[['UID', 'GroupInfo', 'DepthRocs', 'ClassRocs', 'CapRocs']]
        self.depths = d
        self.booking = d[d['ClassRocs'] != 'Св'].groupby(['UID', 'DepthRocs']).sum().\
                         reset_index().rename(columns={'CapRocs': 'Booking'})
        self.booking['DepthRocs'] = fix_str_int_arr(np.array(self.booking['DepthRocs']))
        self.book4 = dict()
        
        for uid, row in self.booking.groupby('UID'):
            a = np.zeros((n_days, 2), dtype=np.int32)
            b = np.array(row, dtype=np.int32)[:, 1:]
            a[:, 0] = np.arange(max_day, min_day - 1, -1)
            a[max_day - b[:, 0], 1] = b[:, 1]
            for i in range(1, len(a)):
                if a[i, 1] == 0:
                    a[i, 1] = a[i-1, 1]
            self.book4[uid] = (a[:, 0], a[:, 1])
        
        
        
        self.rask_dc = pd.read_csv(path.join(datadir, rask_dc_fname), sep=';')
        lb = pd.read_csv(path.join(datadir, labels_fname), sep=';')
        self.labels = dict(zip(lb['uid'], lb['score']))
        self.test_uids = np.array(pd.read_csv(path.join(datadir, test_uids_fname), sep=';')['UID'])
        self.volumes = pd.read_csv(path.join(datadir, volumes_fname), sep=';')
        
        self.rask_dc['RASK'] = fix_str_float_arr(np.array(self.rask_dc['RASK']))
        self.rask_dc['DC'] = fix_str_float_arr(np.array(self.rask_dc['DC']))
        
        for uid, row in self.rask_dc.groupby('UID'):
            a = -np.ones((n_days, 2), dtype=np.float32)
            b = np.array(row[['FlightDate', 'IssueDate', 'DC', 'RASK']])

            for fl_d, is_d, dc, rask in b:
                delta_d = (str2datetime(fl_d) - str2datetime(is_d)).days
                if delta_d < 0:
                    print(uid, fl_d, is_d, dc, rask, delta_d)
                    continue
                idx = max_day - delta_d
                a[idx] = dc, rask
            if a[0, 0] < 0 or a[0, 1] < 0:
                a[0] = 0
                
            for i in range(1, len(a)):
                if a[i, 0] < 0:
                    a[i] = a[i-1]
            self.book4[uid] = (*self.book4[uid], a[:, 0], a[:, 1])

        
        copy_dc = self.rask_dc.copy()
        copy_dc = copy_dc.drop_duplicates(['UID'])
        
        copy_maks = self.labels
        (self.dateCounters, self.dateClick) =  self.create_counters(copy_dc['FlightDate'])
        (self.apCounters, self.apClick) = self.create_counters(copy_dc['FlightRoute'])
        self.ap_list = np.unique(copy_dc['FlightRoute'])
        
        


    def process_series(self, timeline, values, isArima=True):
        #print(timeline.shape)
        #print(len(timeline))
        if (len(timeline) > 10) and isArima:
            mdl = ARIMA(values, dates=timeline, order=(2, 0, 1))
            mdl_fit = mdl.fit()
            params = mdl_fit.params()
        else:
            params = np.zeros(3)
        
        max_val = np.max(values)
        min_val = np.min(values)
        minmax_dif_val = max_val - min_val
        avg_val = np.median(values)
        first_val = values[0]
        last_val = values[-1]
        val_dif = values[1:] - values[:-1]
        median_dif = np.median(val_dif)
        
        #print(timeline[0].timeshtamp())
        ms_timeline = timeline
        print(timeline)
        if (type(ms_timeline[0]) != int):
            ms_timeline = np.array(list(map(lambda x: x.timestamp(), timeline)))
        ms_timeline = ms_timeline[1:] - ms_timeline[:-1]
        min_time = np.min(ms_timeline)
        max_time = np.max(ms_timeline)
        minmax_dif_time = max_time - min_time
        avg_time = np.median(ms_timeline)
        if (type(timeline[0]) != int):
            first_time = timeline[0].timestamp()
            last_time = timeline[-1].timestamp()
        else:
            first_time = timeline[0]
            last_time = timeline[-1]
        
        evr_param = np.array([max_val, min_val, minmax_dif_val, avg_val, first_val, last_val, median_dif, 
                             min_time, max_time, minmax_dif_time, avg_time, first_time, last_time])
        
        return np.hstack((evr_param, params))        
        
    def get_series_features(self, uid):
        data_uid = self.rask_dc[self.rask_dc['UID'] == uid]
        DC_uid = np.array(data_uid['DC'].apply(lambda x: float(x.replace(',', '.'))))
        Dates_uid = list(data_uid['IssueDate'].apply(lambda x: str2datetime(x)))
        dc_params = self.process_series(Dates_uid, DC_uid)
                
        RASK_uid = np.array(data_uid['RASK'].apply(lambda x: float(x.replace(',', '.'))))
        rask_params = self.process_series(Dates_uid, RASK_uid)
        
        book_uid_data = self.booking[self.booking['UID'] == uid]
        book_date = np.array(book_uid_data['DepthRocs'])
        #print(book_date[:10])
        #print(Dates_uid[:10])
        booking_uid = np.array(book_uid_data['Booking'])
        booking_params = self.process_series(book_date, booking_uid, isArima=False)
        
        return np.hstack((dc_params, rask_params, booking_params))
    
    def create_counters(self, feature, clss=None):
        counters = dict([])
        click = dict([])
        for f in np.unique(feature):
            counters[f] = np.sum(feature == f)
            if clss is not None:
                click[f] = np.sum((feature == f) & (clss ==1))/counters[f]
        return counters, click
    
    def get_stand_features_by_uid(self, uid):
        hld = Holiday('get_time_features/holiday.json')
        data_uid = self.rask_dc[self.rask_dc['UID'] == uid]
        #print(uid)
        if len(np.array(data_uid['FlightDate'])) != 0:
            flDate = np.array(data_uid['FlightDate'])[0]
            flight = np.array(data_uid['FlightRoute'])[0]
            dateFeatures = get_time_feature_vector(str2datetime(flDate), hld)
            counters_date = self.dateCounters[flDate]
            counters_flight = self.apCounters[flight]

            vl = self.volumes
            #print(flDate)
            #print(flight)
            vl_uid = (vl[(vl['FlightDate'] == flDate) & (vl['FlightRoute'] == flight)])
            count_uid = np.array(vl_uid['Count'])[0]
            #print(vl_uid)

            return np.hstack((dateFeatures, counters_date, counters_flight, count_uid))
        else:
            return None
        
    def create_train_set(self, isFull=True):
        trainX = np.array([[]])
        trainY = []
        uid = []
        for u, _ in self.labels.items():
            #if self.labels[u] <= 7:
            #    continue
            
            features = self.get_stand_features_by_uid(u)
            
            if features is None:
                print("bad UID", u)
                continue
            trainY.append(self.labels[u])
            uid.append(u)
            
            #print(features)
            if trainX.size == 0:
                graph_featurs = np.array(self.book4[u][1:]).ravel()
                trainX = np.array(np.hstack((graph_featurs, features)))
                #trainX = np.array([features])
            else:
                
                graph_featurs = np.array(self.book4[u][1:]).ravel()
                features = np.hstack((graph_featurs, features))
                #print(trainX.shape, features.shape)
                trainX = np.vstack((trainX, features))
                
        #print(trainX)
        
        trainY = np.array(trainY)
        uid = np.array(uid)
        if not isFull:
            return (trainX[trainY >= 7, :], trainY[trainY >= 7], uid[trainY >= 7])
        
        return (trainX, trainY, uid) 
    
    def create_test_set(self):
        X = np.array([[]])
        for u in self.test_uids:
            features = self.get_stand_features_by_uid(u)    
            
            if features is None:
                print("!!!!")
            #print(features)
            if X.size == 0:
                graph_featurs = np.array(self.book4[u][1:]).ravel()
                X = np.array(np.hstack((features, graph_featurs)))
                #X = np.array([features])
            else:
                
                graph_featurs = np.array(self.book4[u][1:]).ravel()
                features = np.hstack((features, graph_featurs))
                #print(X.shape, features.shape)
                X = np.vstack((X, features))
        return (X, self.test_uids) 
        
        