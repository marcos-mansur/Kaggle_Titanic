# Functions to be used in other files
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
# Learning curve
from sklearn.model_selection import learning_curve

# Model_Selection
from sklearn.model_selection import train_test_split, cross_val_score

#============ Utility functions ===================

@st.cache
def load_data():
    global df, x, y,seed
    df = pd.read_csv("Dados/train.csv",index_col=0)
    x = df.drop("Survived",axis=1).copy()
    y = df.Survived
    seed = 42
    return df,x,y,seed


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes= np.linspace(.1, 1.0, 5)):
    """
    Generate plot: the test and training learning curve
    """
    if axes is None:
        _, axes = plt.subplots(1, figsize=(20, 10))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes, scoring='accuracy',
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes.legend(loc="best")
    print('train_scores_mean: ',train_scores_mean[-1], '\ntest_scores_mean',test_scores_mean[-1])
    return plt

# ========================