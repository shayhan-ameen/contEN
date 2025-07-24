__author__ = 'Shayhan'



# BEST_PATIENTS_RMSE_BELOW_10 = [235, 297, 352, 409, 451, 460, 608, 627, 648, 129, 175, 236, 408, 414, 418, 495, 509, 510, 602, 31, 105, 127, 165, 284, 317,442, 591, 619, 649, 675, 71,
#                                100, 128, 288, 295, 393, 450, 462, 529, 607, 611, 616, 643, 95, 113, 273, 292, 307, 346, 482, 539, 587, 616, 78, 85, 137, 188, 206, 222, 254, 302, 327,
#                                351, 392, 395, 423, 437, 516, 530, 567, 595, 609, 634, 671, 25, 40, 42, 160, 226, 259, 262, 277, 490, 493, 526, 566, 578, 584, 624, 663, 33, 140, 218,
#                                349, 394, 484, 514, 520, 560, 603, 113, 130, 131, 150, 161, 204, 228, 263, 397, 417, 499, 538, 554, 598, 658, 695, 22, 41, 157, 304, 312, 335, 355, 413,
#                                519, 522, 625, 627, 650, 679, 682, 26, 53, 70, 176, 408, 475, 512, 584, 588]

import joblib
from extended_utility import *
from constants import *
from utils import *
from plot_patients import *
# from model import *
from load_data import *
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, GroupKFold
from sklearn.decomposition import PCA
import random
# import torch
import sys
# Modelling
from sklearn import datasets
from seaborn import load_dataset
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2
from sklearn.metrics import mean_squared_error, make_scorer, confusion_matrix, accuracy_score, mean_absolute_error, \
    r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import warnings
import pickle

SEED_VALUE = 99
np.random.seed(SEED_VALUE)
random.seed(SEED_VALUE)
# torch.manual_seed(SEED_VALUE)

sys.path.append(folder_path)

EX_RESULTS = initialize_ex_results()

crf_df = pd.read_pickle(os.path.join(folder_path, 'crf_df.pkl'))
# crf_df = cluster_patients(crf_df, n_clusters=3, sequence_size=D_0_TIME, seed_value=SEED_VALUE)

# patients_data = extract_patients_data(crf_df)
with open(os.path.join(folder_path, 'patients_data.pkl'), 'rb') as f: patients_data = pickle.load(f)
# feeding_data = prepare_feeding_dataset(crf_df, patients_data, SEGMENT, ENS_INTERVAL, extend_min=0, shift=0) # in minutes
with open(os.path.join(folder_path, 'feeding_data.pkl'), 'rb') as f: feeding_data = pickle.load(f)

patients_data = update_patient_feeding_data(patients_data, feeding_data)

pt_list = [pt['index'] for pt in patients_data]
fold=None
errors= []
y_test_pred = []
feeding_list = 1
patient_type = 'with'
messages = None
optimal_insulins = []

# plot_patients_data(output_figures_path, pt_list, patients_data, fold, errors, y_test_pred, feeding_list, D_0_TIME, D_1_TIME, patient_type)

# Plot of patients
# print(len(patients_data))
# plot_patients_data(output_figures_path, pt_list, patients_data, fold, errors, optimal_insulins, feeding_list, D_0_TIME, D_1_TIME, patient_type)

# Plot of patients with feeding
# pt_list = [feed['pt_index'] for feed in feeding_data]
# print(len(pt_list))
# plot_patients_data(output_figures_path, pt_list, patients_data, fold, errors, optimal_insulins, feeding_list, D_0_TIME, D_1_TIME, patient_type)
# simple_plot_single_patient_data



# Plot of patients with feeding
# pt_list =[7, 67, 77, 85, 96, 142, 173, 189, 227, 237, 238, 239, 305, 323, 341, 342, 355, 384, 414, 494, 505, 534, 558, 575, 577, 594, 595, 598, 603, 635, 669, 672, 689, 734, 765, 774, 781, 805, 807, 831, 833]
# feeding_list = [3, 2, 2, 1, 2, 3, 2, 1, 2, 2, 1, 1, 3, 3, 1, 2, 2, 1, 1, 1, 2, 2, 3, 1, 1, 3, 1, 3, 3, 1, 3, 2, 1, 3, 1, 1, 1, 2, 1, 2, 3]
# print(len(pt_list))
# print(len(feeding_list))
# plot_patients_data(output_figures_path, pt_list, patients_data, fold, errors, optimal_insulins, feeding_list, D_0_TIME, D_1_TIME, patient_type)
# simple_plot_single_patient_data
pt_list =[7, 67, 77, 85, 96]
feeding_list = [3, 2, 2, 1]
print(len(pt_list))
print(len(feeding_list))
plot_patients_data(output_figures_path, pt_list, patients_data, fold, errors, optimal_insulins, feeding_list, D_0_TIME, D_1_TIME, patient_type)
# simple_plot_single_patient_data


# MIMIC Dataset
# discarded_patients=[0, 1, 7, 9, 12, 16, 17, 22, 24, 27, 32, 42, 43, 46, 50, 52, 67, 68, 70, 77, 78, 80, 86, 87, 91, 97, 103, 106, 108, 110, 113, 117, 118, 123, 128, 130, 131, 132, 133, 135, 136, 139, 152, 160, 163, 164, 165, 166, 167, 168, 170, 171, 176, 177, 181, 182, 183, 186, 192, 193, 194, 195, 196, 198, 200]
# # pt_list = [pt['index'] for pt in patients_data]
# pt_list = [pt['index'] for pt in patients_data if pt['index'] not in discarded_patients]
# # print(pt_list)
# fold=None
# errors= []
# y_test_pred = []
# feeding_list = 1
# patient_type = 'with'
# messages = None
# optimal_insulins = []
#
# plot_patients_data(output_figures_path, pt_list, patients_data, fold, errors, y_test_pred, feeding_list, D_0_TIME, D_1_TIME, patient_type)