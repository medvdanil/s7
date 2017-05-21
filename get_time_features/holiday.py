import json
from datetime import datetime
import numpy as np

    
class Holiday:
    def __init__(self, json_filename):
        f = open(json_filename)
        data = json.load(f)['data']
        self.min_year = min(list(map(int, data.keys())))
        self.max_year = max(list(map(int, data.keys())))
        self.hdays = -np.ones((self.max_year - self.min_year + 1, 13, 32), dtype=np.int8)
        for y, months in data.items():
            for m, days in months.items():
                for d, val in days.items():
                    self.hdays[int(y) - self.min_year, int(m), int(d)] = val['isWorking']
    
    def isWorking(self, d):
        val = self.hdays[d.year - self.min_year, d.month, d.day]
        return val != 2 if val != -1 else d.weekday() < 5
    
    
def str2datetime(s):
    if s.find(' ') != -1:
        s = s[:s.find(' ')]
    return datetime.strptime(s, '%Y-%m-%d')
