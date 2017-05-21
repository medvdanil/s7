from get_time_features.holiday import Holiday
import datetime 
import numpy as np


def to_next_x(d, hld, direct, day_type):
    dt = datetime.timedelta(days=direct)
    cur_day = d
    to_next = 10
    for i in range(5):
        cur_day = cur_day + dt
        if hld.isWorking(cur_day) == day_type:
            to_next = i
            break
    return to_next   
    
def get_time_feature_vector(d, hld):
    final_vect = np.array([])
    
    #time_features = np.zeros(24)
    #time_features[d.hour] = 1
    #final_vect = np.hstack((final_vect, time_features))
    
    week_day = np.zeros(7)
    week_day[d.weekday()] = 1
    final_vect = np.hstack((final_vect, week_day))
    
    #y_day = datetime.datetime.now().timetuple().tm_yday
    #y_day_feature = np.zeros(366)
    #y_day_feature[y_day] = 1
    #final_vect = np.hstack((final_vect, y_day_feature))
    
    #next_features = np.zeros(4)
    #next_features[0] = to_next_x(d, hld, 1, False)
    #next_features[1] = to_next_x(d, hld, 1, True)
    #next_features[2] = to_next_x(d, hld, -1, False)
    #next_features[3] = to_next_x(d, hld, -1, True)
    #final_vect = np.hstack((final_vect, next_features))
    
    #final_vect = np.hstack((final_vect, [hld.isWorking(d)]))
    
    return final_vect
    
      
    
    