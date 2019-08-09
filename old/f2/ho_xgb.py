
# coding: utf-8
import pandas as pd
import numpy as np
import sklearn 
from tqdm import tqdm_notebook
import catboost
import lightgbm
import xgboost
import eli5
from eli5.sklearn import PermutationImportance

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, Ridge
from sklearn.model_selection import ShuffleSplit, KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

data_train = pd.read_csv('data/data_train_v6_1.csv')
data_test = pd.read_csv('data/data_test_v6_1.csv')
target = data_train.skilled

data_train = data_train.drop(['skilled', 'player_team', 'winner_team','Unnamed: 0' ,'Unnamed: 0.1','Unnamed: 0.1.1', 'Unnamed: 0.1.1.1','Unnamed: 0.1.1.1.1', 'Unnamed: 0.1.1.1.1.1', 'Unnamed: 0.1.1.1.1.1.1','Unnamed: 0.1.1.1.1.1.1.1'], axis=1)
data_test = data_test.drop(['Unnamed: 0','player_team', 'winner_team', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'Unnamed: 0.1.1.1', 'Unnamed: 0.1.1.1.1', 'Unnamed: 0.1.1.1.1.1', 'Unnamed: 0.1.1.1.1.1.1','Unnamed: 0.1.1.1.1.1.1.1'], axis=1)

X = data_train
x_test = data_test
y = target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

def objective(params):
    params = {
        'max_depth': int(params['max_depth']),
        'learning_rate': '{:.3f}'.format(params['learning_rate']),
        'n_estimators': int(params['n_estimators']),
        'min_child_weight': int(params['min_child_weight']),
        'gamma': "{:.3f}".format(params['gamma']),
        'subsample': '{:.3f}'.format(params['subsample']),
        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
        'reg_alpha': '{:.3f}'.format(params['reg_alpha']),
        'objective':  params['objective'],
        'scale_pos_weight': '{:.3f}'.format(params['scale_pos_weight']),
    }
    clf = xgboost.XGBClassifier(
        nthread=4,
        **params
    )
    
    clf.fit(X_train, y_train)
    ac = accuracy_score(y_test, clf.predict(X_test))
    
    f = open('Hyperopt_logs.txt', 'a+')
    f.write("Accuracy of xgb {:.3f} params {}".format(ac, params) + '\n')
    f.close()
    
    print("Accuracy of xgb {:.3f} params {}".format(ac, params))
    
    return ac
    

space = {
    'max_depth': hp.quniform('max_depth', 1, 20, 1),
    'learning_rate':  hp.quniform('learning_rate', 0.001, 0.5, 0.01),
    'n_estimators': hp.quniform('n_estimators', 400, 1500, 100),
    'min_child_weight': hp.quniform('min_child_weight', 1, 20, 1),
    'gamma': hp.quniform('gamma', 0.1, 5, 0.1),
    'subsample': hp.quniform('subsample', 0.3, 0.6, 0.1),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.2, 0.8, 0.1),
    'reg_alpha': hp.quniform('reg_alpha', 0.001, 5, 0.1),
    'objective':  hp.choice('objective', ['binary:logistic', 'binary:logitraw', 'binary:hinge']),
    'scale_pos_weight': hp.quniform('scale_pos_weigh', 0.1, 6, 0.1),
}

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=1000)
