import torch
from torch.utils.data import DataLoader, TensorDataset
from model_offline_rl_ac import *
from extended_utility import *
from constants import *
from utils import *
from plot_patients import *
from graph_generation import *
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
# from xgboost import XGBRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import warnings
import pickle

# MAX_BST_VALUE = 350
# MAX_CALORIE_VALUE = 700
# MAX_INSULIN_VALUE = 26

SEED_VALUE = 99
np.random.seed(SEED_VALUE)
random.seed(SEED_VALUE)
torch.manual_seed(SEED_VALUE)

sys.path.append(folder_path)

EX_RESULTS = initialize_ex_results()

# crf_df = pd.read_pickle(os.path.join(folder_path, 'crf_df.pkl'))
crf_df =merge_datasets('crf_df.pkl', 'mimic_crf_df.pkl')
# crf_df = cluster_patients_crf(crf_df, n_clusters=3, sequence_size=D_0_TIME, seed_value=SEED_VALUE)

patients_data = extract_patients_data(crf_df)
# with open(os.path.join(folder_path, 'patients_data.pkl'), 'rb') as f: patients_data = pickle.load(f)
feeding_data = prepare_feeding_dataset(crf_df, patients_data, SEGMENT, ENS_INTERVAL, extend_min=0*60, shift=0*60) # in minutes
# with open(os.path.join(folder_path, 'feeding_data.pkl'), 'rb') as f: feeding_data = pickle.load(f)
patients_data = update_patient_feeding_data(patients_data, feeding_data)


patient_types, feeding_numbers, bst_ranges = ["without"], [1], [[71,200]]

# discarded_patients =[]
HOSPITAL_LIST = ['DUMC', 'KHNMC', 'KNUH', 'MIMIC-IV']

def condition_func(pt, patient_type, feeding_num, bst_range):
    return (
            (pt['pt_index'] not in  discarded_patients)
            # and pt['initial_bst'] < MAX_BST_VALUE
            # and pt['complete_calorie'] < MAX_CALORIE_VALUE
            # and pt['complete_insulin'] < MAX_INSULIN_VALUE
            # and pt['feeding_num'] == feeding_num
            # and (pt['steroid_sum'] == 0 if patient_type == "without" else pt['steroid_sum'] > 0)
            # and pt['crrt']!=True
            # and (pt['initial_bst'] >= bst_range[0] and pt['initial_bst'] <= bst_range[1])
            # and pt['hospital_name'] in HOSPITAL_LIST
            # and pt['time_in_range'] >=D_0_TIME
            # and pt['gender'] == 0 # 0 - Female, 1 - Male
            # and (pt['cluster'] == 0 if patient_type == "without" else pt['cluster'] == 2)
            # and pt['bst_min_max_dif'] < 100
    )

# features = ['initial_bst', 'extra_insulin', 's_insulin', 'extra_calorie', 's_calorie', 'complete_insulin', 'complete_calorie', 'complete_igr',
#             's_steroid', 'hypo_epis', 'hyper_epis', 'time_in_range', 'duration', 'bst_mean_difference', 'insulin_mean_difference',
#             'calorie_mean_difference', 'bst_min_nz', 'bst_max', 'bst_min_max_dif', 'combine_w_ci', 'initial_bst_slope', 's_insulin_sum', 's_calorie_sum']
features = ['initial_bst', 'complete_calorie', 'fasting_igr_level', 'age', 'gender', 'hgt', 'bwt']
# features = ['initial_bst', 'complete_calorie', 'fasting_igr_level', 'age', 'gender', 'bwt']
# features = ['initial_bst', 'complete_calorie', 'fasting_igr_level']
# features = ['initial_bst', 'complete_calorie']

INS_DIFS = []
BST_DIFS = []

patient_type, feeding_num, bst_range = "without", 1, [71,200]

all_patient_graph_list, discarded_patients = create_all_patient_graphs(crf_df, patients_data, D_0_TIME, min_num_nodes=3)
# print(f"{discarded_patients=}")


X, y, insulins_administrated, flt_pts, flt_feed, flt_pts_cluster = filter_patients(feeding_data, features, SEGMENT, lambda pt: condition_func(pt, patient_type, feeding_num, bst_range))

# Split dataset based on hospital names
train_idx = [i for i, _ in enumerate(flt_pts_cluster) if flt_pts_cluster[i] not in ['MIMIC-IV']]
test_idx = [i for i, _ in enumerate(flt_pts_cluster) if flt_pts_cluster[i] in ['MIMIC-IV']]
# train_idx = [i for i, pt in enumerate(flt_pts) if flt_pts_cluster[i] not in ['MIMIC-IV']]
# test_idx = [i for i, pt in enumerate(flt_pts) if flt_pts_cluster[i] in ['MIMIC-IV']]

# Convert lists to numpy arrays for indexing
train_idx = np.array(train_idx)
test_idx = np.array(test_idx)

print(f"{len(train_idx)=}")
print(f"{len(test_idx)=}")
print(f"Total: {len(X)}")
# print(flt_pts_cluster[train_idx])
# X= filter_nan_rows(X, flt_pts)
# if X is None: continue

# print(f"Total: {len(np.unique(flt_pts))}")


rewards = reward_calculation(y, desired_bst = 130)
# rewards = [r for i, r in enumerate(rewards) if i not in discarded_patients]
states, actions = X, insulins_administrated
max_action = max(actions)

scaler = StandardScaler()
states = scaler.fit_transform(states) # Fit the scaler on the training data and transform both train and test sets

# # Generate Graph Embeddings
# graph_embeddings = generate_graph(crf_df, patients_data, D_0_TIME)
# graph_embeddings = graph_embeddings.to(torch.float32)  # Ensure dtype compatibility
# # Concatenate Graph Embeddings to RL states
# combined_states = torch.cat((states, graph_embeddings), dim=1)  # New state representation
# # Update state dimension for RL Model
# state_dim = combined_states.shape[1]
# agent = td3_bc(max_action, state_dim)

# Ensure they are PyTorch tensors
states = torch.tensor(states, dtype=torch.float32)
states = torch.tensor(states, dtype=torch.float32)
actions = torch.tensor(actions, dtype=torch.float32)
rewards = torch.tensor(rewards, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# group_kfold = GroupKFold(n_splits=len(np.unique(flt_pts)))
group_kfold = GroupKFold(n_splits=10)
group_kfold.get_n_splits(X, y, flt_pts)

# Initialize accumulators
total_bgl_mae = 0.0
total_ins_acc_correct = 0
total_ins_acc_count = 0
num_folds = 0

EPOCHS = 30
print("---------------------------")
print(f"EPOCHS = {EPOCHS}")
print("---------------------------")
patient_feeding_insulin_info = []
# for fold, (train_idx, test_idx) in enumerate(group_kfold.split(X, y, flt_pts)):
    # print(f"{np.unique(flt_pts[train_idx])=}")
    # print(f"{np.unique(flt_pts[test_idx])=}")
    # break
    # if fold >3: continue
    # if fold not in [1, 5, 177, 221, 139]: continue
num_folds += 1

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
insulins_administrated_train, insulins_administrated_test = insulins_administrated[train_idx], insulins_administrated[test_idx]
flt_pts_train, flt_pts_test = flt_pts[train_idx], flt_pts[test_idx]
flt_pts_cluster_test = flt_pts_cluster[test_idx]
states_train, states_test = states[train_idx], states[test_idx]
ax_states_train, ax_states_test = states_train[:, 2:], states_test[:, 2:]
states_train, states_test = states_train[:, :2], states_test[:, :2]
flt_feed_train = [flt_feed[idx] for idx in train_idx]
flt_feed_test = [flt_feed[idx] for idx in test_idx]
# print(f"{X_train.shape=}")
# print(f"{X_test.shape=}")
# print(f"{states_train[:5]}")

actions_train, actions_test = actions[train_idx], actions[test_idx]
rewards_train, rewards_test = rewards[train_idx], rewards[test_idx]
actions_train = actions_train.unsqueeze(1)
actions_test = actions_test.unsqueeze(1)
rewards_train = rewards_train.unsqueeze(1)
rewards_test = rewards_test.unsqueeze(1)
flt_pts_train_tensor = torch.tensor(flt_pts_train, dtype=torch.float32).unsqueeze(1)
flt_pts_test_tensor = torch.tensor(flt_pts_test, dtype=torch.float32).unsqueeze(1)
y_train = y_train.unsqueeze(1)

dataset_train = TensorDataset(states_train, actions_train, rewards_train, flt_pts_train_tensor, y_train, ax_states_train)
dataset_test = TensorDataset(states_test, actions_test, rewards_test, flt_pts_test_tensor, ax_states_test)
dataloader_test = DataLoader(dataset_test, batch_size=1000, shuffle=False)

# agent = td3_bc(max_action, X_train.shape[1])
# agent = td3_bc(max_action, num_rl_in_dim=states_train.shape[1], num_graph_in_dim=5, num_graph_out_dim=8, all_patient_graph_list = all_patient_graph_list, hidden_dim=128, action_size=1)
agent = td3_bc(max_action, rl_in_dim=states_train.shape[1], graph_in_dim=5, graph_hidden_dim=128, graph_out_dim=8, ax_in_dim=ax_states_train.shape[1], ax_out_dim=8, pf_out_dim=8, all_patient_graph_list = all_patient_graph_list, action_size=1)

agent.train_model(dataset_train, EPOCHS ) # Train the agent

predicted_insulin = agent.select_action(dataloader_test, max_action) # Test the agent

for i in range(len(insulins_administrated_test)):
    # BST_DIFS.append(X_test[i, 0])
    BST_DIFS.append(y_test[i])
    INS_DIFS.append(predicted_insulin[i] - insulins_administrated_test[i])
    patient_feeding_insulin = {
        "pt_index": flt_feed_test[i]["pt_index"],
        "hospital_name": flt_feed_test[i]["hospital_name"],
        "hospital_num": flt_feed_test[i]["hospital_num"],
        "feeding_num": flt_feed_test[i]["feeding_num"],
        "pt_num_feed": f'{flt_feed_test[i]["hospital_num"]}_{flt_feed_test[i]["feeding_num"]}',
        "BST_diff": X_test[i, 0]-130,
        "INS_diff": predicted_insulin[i] - insulins_administrated_test[i],
        "actual_insulin": insulins_administrated_test[i],
        "predicted_insulin": predicted_insulin[i],
    }
    patient_feeding_insulin_info.append(patient_feeding_insulin)

# print(f"{fold} complete.")

plot_bst_vs_ins_quadrant(BST_DIFS, INS_DIFS, "all", EPOCHS)

save_path = "results/main graph/current 898"
with open(os.path.join(save_path, 'current_patient_feeding_insulin_info.pkl'), 'wb') as f:
    pickle.dump(patient_feeding_insulin_info, f)



