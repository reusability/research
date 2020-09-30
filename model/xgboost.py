from model.base_model import BaseModel
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
import pandas as pd
import numpy as np 
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

class XGBoost(BaseModel):
    def __init__(self, data, normalize=False):
        BaseModel.__init__(self, data, normalize=normalize)

        self.model = XGBClassifier()

    def train(self):
        print('Training myself ahahhaha')
    
    