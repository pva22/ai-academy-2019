
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

train = pd.read_csv('data/trian_v2.csv', index_col='id')
test = pd.read_csv('data/test_v2.csv', index_col='id')
target = train.skilled
train = train.drop(['skilled'], axis=1)
X = train
x_test = test
y = target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

def objective(params):
    params = {
        'max_depth': int(params['max_depth']),
        'learning_rate': '{:.3f}'.format(params['learning_rate']),
        'min_child_weight': int(params['min_child_weight']),
        'reg_alpha': '{:.3f}'.format(params['reg_alpha']),
    }
    clf = lightgbm.LGBMClassifier(
        n_estimators = 1500,
        nthread=-1,
        **params
    )
    
    clf.fit(X_train, y_train)
    ac = accuracy_score(y_test, clf.predict(X_test))
    
    f = open('Hyperopt_logs_lgb.txt', 'a+')
    f.write("Accuracy of lgb {:.3f} params {}".format(ac, params) + '\n')
    f.close()
    
    print("Accuracy of lgb {:.3f} params {}".format(ac, params))
    
    return ac
    

space = {
    'max_depth': hp.quniform('max_depth', 6, 20, 1),
    'learning_rate':  hp.quniform('learning_rate', 0.01, 0.8, 0.01),
    'min_child_weight': hp.quniform('min_child_weight', 5, 20, 1),
    'reg_alpha': hp.quniform('reg_alpha', 5, 15, 1)
}

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=1000)
