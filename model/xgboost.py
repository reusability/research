from model.base_model import BaseModel
import xgboost as xgb
from xgboost import DMatrix
from xgboost import train
import pandas as pd
import numpy as np 
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

class XGBoost(BaseModel):
    def __init__(self, data, normalize=False):
        BaseModel.__init__(self, data, normalize=normalize)

        #self.model = xgb.XGBClassifier()
    
    def train(self):
        train_xgb = DMatrix(self.train_x, label=self.train_y)

        params = {
            'max_depth': None,  # the maximum depth of each tree
            'eta': None,  # the training step for each iteration
            'silent': None,  # logging mode - quiet
            'objective': None,  # error evaluation for multiclass training
            'num_class': None,
            'n_estimators': None
        }

        self.m = train(params, train_xgb, num_boost_round=10)
    
    def test(self):
        test_xgb = DMatrix(self.test_x)
        res = self.m.predict(test_xgb)




