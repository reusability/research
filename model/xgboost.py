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
    
    def modelfit(self, algorithm, dtrain, predictors, target, useTrainCV=True, folds=5, early_stopping_rounds=50):
        """References
        https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
        """
        if useTrainCV is True:
            xgb_params = algorithm.get_xgb_params()
            xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
            cv_result = xgb.cv(xgb_params, xgtrain, num_boost_round=algorithm.get_params()['n_estimators'], nfold=folds, metrics='auc', early_stopping_rounds=early_stopping_rounds, show_progress=False)
            algorithm.set_params(n_estimators=cv_result.shape[0])
        algorithm.fit(dtrain[predictors], dtrain[target], eval_metrics='auc')

        dtrain_predictions = algorithm.predict(dtrain[predictors])
        dtrain_predprob = algorithm.predict_proba(dtrain[predictors])[:,1]
        feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')


    def train(self):
        train_xgb = xgb.DMatrix(self.train_x, label=self.train_y)
        

        params = {
            'max_depth': 4,  # the maximum depth of each tree
            'eta': 0.3,  # the training step for each iteration
            'silent': 1,  # logging mode - quiet
            'objective': 'multi:softprob',  # error evaluation for multiclass training
            'num_class': 3,
            'n_estimators': 10
        }

        CROSS_VAL = True
        if CROSS_VAL:
            print('Doing Cross-validation ...')
            cv = xgb.cv(params, train_xgb, nfold=3, early_stopping_rounds=10, metrics='mlogloss', verbose_eval=True)
            print(cv)

        print('Fitting Model ...')
        self.m = xgb.train(params, train_xgb, num_boost_round=10)
  
    def test(self):
        res = self.m.predict(test_xgb)

    def hyperparameter_tuning(self):