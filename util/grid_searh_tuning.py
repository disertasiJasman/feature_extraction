import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier


def get_best_param(build_fn, param_grid, X_train, Y_train):

    model = KerasClassifier(build_fn=build_fn,
                            epochs=10,
                            batch_size=5,
                            verbose=0)
    # create_model().summary()

    """
    param_grid = {'dense_layers': [[4], [8], [8, 8]],
                  'activation': ['relu', 'tanh'],
                  'optimizer': ('rmsprop', 'adam'),
                  'epochs': [10, 50],
                  'batch_size': [5, 16]}
    """

    grid = GridSearchCV(model,
                        param_grid=param_grid,
                        return_train_score=True,
                        scoring=['precision_macro', 'recall_macro', 'f1_macro'],
                        refit='precision_macro')

    grid_results = grid.fit(X_train, Y_train)

    print('Parameters of the best model: ')
    print(grid_results.best_params_)

    return grid_results.best_params_