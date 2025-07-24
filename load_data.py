__author__ = 'Shayhan'

from utils import *

import pandas as pd

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np


def process_feeding_data(feeding_data):
    feeding_df = pd.DataFrame(feeding_data)
    df = feeding_df.copy()
    mask = (df['complete_insulin'] > 0) & (df['complete_calorie'] > 0) & (df['duration'] > 0)
    df = df[mask].copy()

    df['delta_bgl'] = df['initial_bst'] - df['target_bst']
    df['duration_hr'] = df['duration'] / 60
    df['IS_T'] = (np.abs(df['delta_bgl']) * df['complete_calorie']) / (df['complete_insulin'] * df['duration_hr'])
    df['IS_T'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['IS_T'], inplace=True)

    df_bgl = {
        'Low_up': df[(df['target_bst'] < 80) & (df['delta_bgl'] > 0)],
        'Low_down': df[(df['target_bst'] < 80) & (df['delta_bgl'] <= 0)],
        'Normal_up': df[(df['target_bst'] >= 80) & (df['target_bst'] <= 180) & (df['delta_bgl'] > 0)],
        'Normal_down': df[(df['target_bst'] >= 80) & (df['target_bst'] <= 180) & (df['delta_bgl'] <= 0)],
        'High_up': df[(df['target_bst'] > 180) & (df['delta_bgl'] > 0)],
        'High_down': df[(df['target_bst'] > 180) & (df['delta_bgl'] <= 0)]
    }

    group_stats = {}
    col = 'IS_T'
    for group, df_group in df_bgl.items():
        q_low = df_group[col].quantile(0.1)
        q_high = df_group[col].quantile(0.90)
        # if "Low" in group:
        #     df_filt = df_group.copy()
        # else:
        #     df_filt = df_group[(df_group[col] >= q_low) & (df_group[col] <= q_high)].copy()
        df_filt = df_group[(df_group[col] >= q_low) & (df_group[col] <= q_high)].copy()
        mu, sigma = df_filt[col].mean(), df_filt[col].std()
        IS_minus = df_filt[col].quantile(0.165)
        IS_plus = df_filt[col].quantile(0.835)
        group_stats[group] = {"IS_mean": mu, "sigma": sigma, "IS_minus": IS_minus, "IS_plus": IS_plus}

    def compute_insulin_bounds(df, IS_minus, IS_mean, IS_plus):
        df = df.copy()
        df['delta_bgl'] = df['initial_bst'] - df['target_bst']
        df['duration_hr'] = df['duration'] / 60
        df = df[(df['duration_hr'] > 0) & (df['complete_calorie'] > 0)]
        df['IN_upper'] = (np.abs(df['delta_bgl']) * df['complete_calorie']) / (IS_minus * df['duration_hr'])
        df['IN_mean'] = (np.abs(df['delta_bgl']) * df['complete_calorie']) / (IS_mean * df['duration_hr'])
        df['IN_lower'] = (np.abs(df['delta_bgl']) * df['complete_calorie']) / (IS_plus * df['duration_hr'])
        df['IN_delta'] = (np.abs(df['IN_upper']) - df['IN_lower']) / 2
        df['IN_upper_hr'] = (df['IN_upper'] * 60) / df['duration']
        df['IN_mean_hr'] = (df['IN_mean'] * 60) / df['duration']
        df['IN_lower_hr'] = (df['IN_lower'] * 60) / df['duration']
        df['IN_delta_hr'] = (np.abs(df['IN_upper_hr']) - df['IN_lower_hr']) / 2
        return df

    df_bgl_calc = {}
    for group, df_group in df_bgl.items():
        stats = group_stats[group]
        df_calc = compute_insulin_bounds(df_group, stats['IS_minus'], stats['IS_mean'], stats['IS_plus'])
        df_bgl_calc[group] = df_calc
        group_stats[group].update({
            "mean_IN_upper": df_calc["IN_upper"].mean(),
            "mean_IN_mean": df_calc["IN_mean"].mean(),
            "mean_IN_lower": df_calc["IN_lower"].mean(),
            "mean_IN_delta": df_calc["IN_delta"].mean(),
            "mean_IN_upper_hr": df_calc["IN_upper_hr"].mean(),
            "mean_IN_mean_hr": df_calc["IN_mean_hr"].mean(),
            "mean_IN_lower_hr": df_calc["IN_lower_hr"].mean(),
            "mean_IN_delta_hr": df_calc["IN_delta_hr"].mean()
        })

    cols_to_merge = ['pt_index', 'feeding_num', 'IN_upper', 'IN_mean', 'IN_lower', 'IN_delta',
                     'IN_lower_hr', 'IN_upper_hr', 'IN_mean_hr', 'IN_delta_hr']
    merged_data = [df_group[cols_to_merge].copy() for df_group in df_bgl_calc.values()
                   if all(col in df_group.columns for col in cols_to_merge)]
    if merged_data:
        df_merge_ready = pd.concat(merged_data, ignore_index=True)
        feeding_df = feeding_df.merge(df_merge_ready, on=['pt_index', 'feeding_num'], how='left')

    group_stats['Low_down'] = group_stats['Low_up']
    df = feeding_df.copy()
    df['delta_bgl'] = df['initial_bst'] - df['target_bst']

    # Grouping by condition + NaNs in IN_upper
    df_bgl = {
        'Low_up': df[(df['target_bst'] < 80) & (df['delta_bgl'] > 0) & (df['IN_upper'].isna())],
        'Low_down': df[(df['target_bst'] < 80) & (df['delta_bgl'] <= 0) & (df['IN_upper'].isna())],
        'Normal_up': df[
            (df['target_bst'] >= 80) & (df['target_bst'] <= 180) & (df['delta_bgl'] > 0) & (df['IN_upper'].isna())],
        'Normal_down': df[
            (df['target_bst'] >= 80) & (df['target_bst'] <= 180) & (df['delta_bgl'] <= 0) & (df['IN_upper'].isna())],
        'High_up': df[(df['target_bst'] > 180) & (df['delta_bgl'] > 0) & (df['IN_upper'].isna())],
        'High_down': df[(df['target_bst'] > 180) & (df['delta_bgl'] <= 0) & (df['IN_upper'].isna())]
    }

    # Fill NaNs in-place
    for group, df_group in df_bgl.items():
        stats = group_stats[group]
        mask = df_group.index  # Index of rows to update

        df.loc[mask, 'IN_upper'] = stats['mean_IN_upper']
        df.loc[mask, 'IN_mean'] = stats['mean_IN_mean']
        df.loc[mask, 'IN_lower'] = stats['mean_IN_lower']
        df.loc[mask, 'IN_delta'] = stats['mean_IN_delta']
        df.loc[mask, 'IN_upper_hr'] = stats['mean_IN_upper_hr']
        df.loc[mask, 'IN_mean_hr'] = stats['mean_IN_mean_hr']
        df.loc[mask, 'IN_lower_hr'] = stats['mean_IN_lower_hr']
        df.loc[mask, 'IN_delta_hr'] = stats['mean_IN_delta_hr']

    df['complete_insulin_hr'] = (df['complete_insulin'] * 60) / df['duration']

    # Clamp IN_lower_hr and IN_upper_hr based on complete_insulin_hr
    mask_lower = df['complete_insulin_hr'] < df['IN_lower']
    df.loc[mask_lower, 'IN_lower'] = (
                df.loc[mask_lower, 'complete_insulin_hr'] - df.loc[mask_lower, 'IN_delta']).clip(lower=0)

    mask_upper = df['complete_insulin_hr'] > df['IN_upper']
    df.loc[mask_upper, 'IN_upper'] = (df.loc[mask_upper, 'complete_insulin_hr'] + df.loc[mask_upper, 'IN_delta'])

    return df, group_stats


def merge_datasets(org_file_path, mimic_file_path, merged_file_path=None):
    """
    Merges mimic_crf_df into org_crf_df while:
    - Keeping all columns from org_crf_df
    - Ignoring extra columns in mimic_crf_df
    - Appending mimic_crf_df at the end of org_crf_df
    - Dropping fully empty columns to prevent FutureWarning
    - Saving the merged dataset to merged_file_path

    Parameters:
    org_file_path (str): Path to the original dataset (org_crf_df)
    mimic_file_path (str): Path to the MIMIC dataset (mimic_crf_df)
    merged_file_path (str): Path to save the merged dataset

    Returns:
    pd.DataFrame: The merged DataFrame
    """

    # Load the datasets
    org_crf_df = pd.read_pickle(os.path.join(folder_path, org_file_path))
    mimic_crf_df = pd.read_pickle(os.path.join(folder_path, mimic_file_path))

    # Step 1: Identify common columns (ignore extra columns in mimic_crf_df)
    common_cols = [col for col in mimic_crf_df.columns if col in org_crf_df.columns]

    # Step 2: Select only the common columns from mimic_crf_df
    mimic_crf_df = mimic_crf_df[common_cols]

    # Step 3: Add any missing columns from org_crf_df (fill with NaN)
    # for col in org_crf_df.columns:
    #     if col not in mimic_crf_df.columns:
    #         # print(f"Missing ---> {col=}")
    #         mimic_crf_df[col] = pd.NA

    # Identify missing columns
    missing_cols = [col for col in org_crf_df.columns if col not in mimic_crf_df.columns]
    # Create a new DataFrame with missing columns filled with NaN
    missing_df = pd.DataFrame(pd.NA, index=mimic_crf_df.index, columns=missing_cols)
    # Concatenate along columns (axis=1) to avoid fragmentation
    mimic_crf_df = pd.concat([mimic_crf_df, missing_df], axis=1)

    # Step 4: Reorder mimic_crf_df to match org_crf_df column order
    mimic_crf_df = mimic_crf_df[org_crf_df.columns]

    # 🔥 FIX: Drop completely empty (all-NA) columns before merging
    mimic_crf_df = mimic_crf_df.dropna(axis=1, how='all')

    # Step 5: Append mimic_crf_df at the end of org_crf_df
    merged_df = pd.concat([org_crf_df, mimic_crf_df], ignore_index=True, sort=False)

    # Step 6: Save the merged DataFrame
    if merged_file_path:
        merged_df.to_pickle(merged_file_path)

    # Print confirmation
    # print("✅ Merging complete. Merged DataFrame saved at:", merged_file_path)
    print("📊 Merging complete. Final DataFrame Shape:", merged_df.shape)

    return merged_df  # Return the merged DataFrame

    return merged_df

def calculate_ic_consumption(start_idx, end_idx, insulin, w_i, c_i, calorie, w_c, c_c,
                             pre_window=IC_WINDOW - SEG_WINDOW, sensitivity_use=True):
    insulin_portion = np.array(insulin[start_idx - pre_window:end_idx])
    calorie_portion = np.array(calorie[start_idx - pre_window:end_idx])
    try:
        reshaped_insulin = insulin_portion.reshape(16, 15)
        reshaped_calorie = calorie_portion.reshape(16, 15)
    except ValueError:
        raise ValueError(
            "The size of calories vector is not divisible by 16 with portion size 15.")
    insulin_sums = reshaped_insulin.sum(axis=1)
    calorie_sums = reshaped_calorie.sum(axis=1)
    insulin_consumption = c_i * np.dot(w_i, insulin_sums) if sensitivity_use else np.dot(w_i, insulin_sums)
    calorie_consumption = c_c * np.dot(w_c, calorie_sums) if sensitivity_use else np.dot(w_c, calorie_sums)
    return insulin_consumption, calorie_consumption


def update_patient_feeding_data(patients_data, feeding_data):
    """
    Update the feeding data for patients.

    Parameters:
    - patients_data (dict): Dictionary containing patient data.
    - feeding_data (list): List of feeding records with index, feeding_num, s_insulin, and s_calorie.

    Returns:
    - Updated patients_data with feeding data.
    """
    for feeding in feeding_data:
        feeding_num = feeding['feeding_num']
        patient_index = feeding['pt_index']

        patients_data[patient_index][f"feeding_{feeding_num}"] = {
            # 'seg_insulin': feeding['s_insulin'],
            # 'seg_calorie': feeding['s_calorie'],
            'complete_calorie': feeding['complete_calorie'],
            'complete_insulin': feeding['complete_insulin'],
            'target_bst_idx': feeding['target_bst_idx'],
            'target_bst': feeding['target_bst'],
            'initial_bst_idx': feeding['initial_bst_idx'],
            'initial_bst': feeding['initial_bst'],
            # 'segment_idx': feeding['segment_idx'],
            'duration': feeding['duration']
        }
    return patients_data


def load_patient_data(pt, pt_row, bst_sample, bst_inter_sample, bst_inter_agg_semple, insulin_sample, steroid_sample,
                      calorie_sample, tf_sample, ens_sample, ens_distributed_sample, medication_sample, insulin_i_sample,
                      insulin_h_sample, insulin_i_iv_sample): #calorie_a_sample, insulin_a_sample, pt_ic_window, crrt, hd,
    """
    Load individual patient data into a structured dictionary.
    Slices the arrays inside the function to avoid repeated memory allocation.
    """
    # lower_bound_bst, upper_bound_bst = 100, 180
    lower_bound_bst, upper_bound_bst = 70, 200

    d0endHours = 25

    fasting_insulin = insulin_sample[:D_0_TIME].sum()
    fasting_calorie = calorie_sample[:D_0_TIME].sum()
    fasting_igr = fasting_insulin / (fasting_calorie + 0.01)
    # hypo_epis = np.sum(bst_inter_agg_semple[:d0endHours] < lower_bound_bst)
    # hyper_epis = np.sum(bst_inter_agg_semple[:d0endHours] > upper_bound_bst)
    # time_in_range = np.sum((bst_inter_agg_semple[:d0endHours] >= lower_bound_bst) & (bst_inter_agg_semple[:d0endHours] <= upper_bound_bst))
    hypo_epis = np.sum(bst_inter_sample[:D_0_TIME] < lower_bound_bst)
    hyper_epis = np.sum(bst_inter_sample[:D_0_TIME] > upper_bound_bst)
    time_in_range = np.sum((bst_inter_sample[:D_0_TIME] >= lower_bound_bst) & (bst_inter_sample[:D_0_TIME] <= upper_bound_bst))

    bst_min_nz = np.min(bst_sample[:D_0_TIME][bst_sample[:D_0_TIME] > 0]) if np.any(bst_sample[:D_0_TIME]> 0) else 0
    bst_max = np.max(bst_sample[:D_0_TIME])
    bst_min_max_dif = bst_max-bst_min_nz

    pre_EN_total_insulin = insulin_sample[:D_0_TIME].sum()
    post_EN_total_insulin = insulin_sample[D_0_TIME:D_1_TIME].sum()
    pre_EN_total_calorie = calorie_sample[:D_0_TIME].sum()
    # post_EN_total_calorie = calorie_sample[D_0_TIME:D_1_TIME].sum()
    pre_EN_agv_bst = bst_sample[:D_0_TIME][bst_sample[:D_0_TIME] != 0].mean()
    post_EN_agv_bst = bst_sample[D_0_TIME:D_1_TIME][bst_sample[D_0_TIME:D_1_TIME] != 0].mean()
    EN_total_calorie = ens_sample[D_0_TIME:D_1_TIME].sum()
    post_EN_total_calorie = calorie_sample[D_0_TIME:D_1_TIME].sum() - ens_sample[D_0_TIME:D_1_TIME].sum()



    # Construct patient data dictionary
    pt_data = {
        'index': pt,
        'hospital_name': pt_row['Hospi.name'],
        'hospital_num': pt_row['Hospi.No'],
        'pt_num': pt_row['Hospi.No'],
        'feeding_start_time': pt_row['feeding_start_time'],
        'age': pt_row['Age'],
        'gender': 1 if pt_row['Gender'] == 'M' else 0,
        'hgt': pt_row['Ht'],
        'bwt': pt_row['Bwt'],
        'bmi': pt_row['Bwt'] / pt_row['Ht'],
        # 'insulin_mean': pt_ic_window['pt_mean_i'],
        # 'insulin_std': pt_ic_window['pt_std_i'],
        # 'insulin_sensitivity': pt_ic_window['c_i'],
        # 'calorie_mean': pt_ic_window['pt_mean_c'],
        # 'calorie_std': pt_ic_window['pt_std_c'],
        # 'calorie_sensitivity': pt_ic_window['c_c'],
        'bst_min_nz': bst_min_nz,
        'bst_max': bst_max,
        'bst_min_max_dif': bst_min_max_dif,
        'fasting_insulin': fasting_insulin,
        'fasting_calorie': fasting_calorie,
        'fasting_igr': fasting_igr,
        'hypo_epis': hypo_epis,
        'hyper_epis': hyper_epis,
        'time_in_range': time_in_range,
        # 'crrt': crrt,
        # 'hd': hd,
        # Temporal Data
        'bst': bst_sample,
        'bst_inter': bst_inter_sample,
        'insulin': insulin_sample,
        'steroid': steroid_sample,
        'calorie': calorie_sample,
        # 'calorie_a': calorie_a_sample,
        'tpn': tf_sample,
        'ens': ens_sample,
        'ens_distributed': ens_distributed_sample,
        'medication': medication_sample,
        'insulin_i': insulin_i_sample,
        'insulin_h': insulin_h_sample,
        'insulin_iv': insulin_i_iv_sample,
        # 'insulin_a': insulin_a_sample,
        #--------------
        "pre_EN_total_insulin": pre_EN_total_insulin,
        "post_EN_total_insulin": post_EN_total_insulin,
        "pre_EN_total_calorie": pre_EN_total_calorie,
        "post_EN_total_calorie": post_EN_total_calorie,
        "pre_EN_agv_bst": pre_EN_agv_bst,
        "post_EN_agv_bst": post_EN_agv_bst,
        "EN_total_calorie": EN_total_calorie,
        # Merge feeding data dynamically
        **{f'feeding_{i + 1}': {} for i in range(MAX_FEEDING)},
    }
    return pt_data


def extract_patients_data(crf_df):
    """
    Extracts all patients' data from the CRF dataframe and store it in a structured format.
    """
    # with open(os.path.join(folder_path, 'patients_ic_window.pkl'), 'rb') as f:
    #     patients_ic_window = pickle.load(f)

    # Convert each column from the DataFrame to a NumPy array and replace NaNs with 0
    # samples_list = ['bst_sequence', 'bst_sequence_inter', 'insulin', 'calorie', 'calorie_a', 'ens',
    #                 'ens_distributed', 'insulin_i', 'insulin_h', 'insulin_i_iv', 'insulin_a', 'steroid']

    samples_list = ['bst_sequence', 'bst_sequence_inter', 'insulin', 'calorie', 'ens',
                    'ens_distributed', 'insulin_i', 'insulin_h', 'insulin_i_iv', 'steroid']

    # Apply np.nan_to_num() for each array
    bst_samples, bst_inter_samples, insulin_samples, calorie_samples, ens_samples, ens_distributed_samples, \
    insulin_i_samples, insulin_h_samples, insulin_i_iv_samples, steroid_samples = \
        [np.nan_to_num(np.array(crf_df[col].tolist())) for col in samples_list]

    # crf_df['crrt'] = crf_df.apply(lambda row: not (pd.isna(row['crrt_start_c_1']) or row['crrt_start_c_1'] == '0'), axis=1)
    hd_start_cols = [x for x in crf_df.columns if "HD_start" in x]
    # crf_df['hd'] = crf_df[hd_start_cols].apply(lambda row: not row.isna().any(), axis=1)
    # crrt = np.array(crf_df['crrt'].tolist())
    # hd = np.array(crf_df['hd'].tolist())

    bst_inter_agg_semples = aggregate_minutes(bst_inter_samples, period=60, method='mean')

    # Calculate the TPN/Fluid values
    tf_samples = calorie_samples - ens_distributed_samples

    # Convert medication values to binary once
    medication_samples = (np.array(crf_df['medication'].tolist()) > 0).astype(int)

    # Parallelize the patient data extraction
    patients_data = Parallel(n_jobs=-1)(
        delayed(load_patient_data)(
            pt, crf_df.loc[pt], bst_samples[pt], bst_inter_samples[pt], bst_inter_agg_semples[pt], insulin_samples[pt],
            steroid_samples[pt], calorie_samples[pt], tf_samples[pt], ens_samples[pt], ens_distributed_samples[pt],
            medication_samples[pt], insulin_i_samples[pt], insulin_h_samples[pt], insulin_i_iv_samples[pt]
            # calorie_a_samples[pt], insulin_a_samples[pt], patients_ic_window[pt] # crrt[pt], hd[pt],
        ) for pt in crf_df.index
    )

    clustering_features = ['bst_min_max_dif', 'time_in_range', 'bmi']
    patients_data = cluster_patients_data(patients_data, clustering_features)

    # Plot 2D and 3D visualizations
    # plot_clusters_2d_3d(patients_data, features=clustering_features)

    # Saving patients_data to a .pkl file
    with open(os.path.join(folder_path, 'patients_data.pkl'), 'wb') as f:
        pickle.dump(patients_data, f)

    return patients_data


def cluster_patients_data(patients_data, clustering_features, n_clusters=3, save_path=None):
    """
    Clusters patients based on the specified features and adds cluster labels to the data.

    Args:
        patients_data (list of dict): List of patient data dictionaries.
        clustering_features (list of str): List of features to use for clustering.
        n_clusters (int): Number of clusters to form. Default is 3.
        save_path (str): Path to save the updated patients_data with clusters. Default is None.

    Returns:
        list of dict: Updated patients_data with cluster labels.
    """
    # Convert patients_data to a DataFrame
    patients_df = pd.DataFrame(patients_data)

    # Select features for clustering
    clustering_data = patients_df[clustering_features]

    clustering_data = clustering_data.apply(lambda x: x.fillna(x.mean()), axis=0)

    # Standardize the data
    scaler = StandardScaler()
    clustering_data_scaled = scaler.fit_transform(clustering_data)

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=99)
    patients_df['cluster'] = kmeans.fit_predict(clustering_data_scaled)

    # Count the number of data points in each cluster
    cluster_counts = patients_df['cluster'].value_counts()

    # Display the counts
    # print(cluster_counts)

    # Add cluster labels back to patients_data
    for i, patient in enumerate(patients_data):
        patient['cluster'] = patients_df.loc[i, 'cluster']

    # # Save updated data if save_path is provided
    # if save_path:
    #     with open(save_path, 'wb') as f:
    #         pickle.dump(patients_data, f)

    return patients_data


def plot_clusters_2d_3d(patients_data, features, cluster_col='cluster'):
    """
    Plots 2D and 3D visualizations of the clustered patients data.

    Args:
        patients_data (list of dict): List of patient data with cluster labels.
        features (list of str): Features to use for the plots.
        cluster_col (str): Column name for cluster labels.
    """
    # Convert patients_data to a DataFrame
    patients_df = pd.DataFrame(patients_data)

    # Ensure the required features are present
    if len(features) < 3:
        raise ValueError("At least 3 features are needed for 3D visualization.")

    # 2D Scatter Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=patients_df,
        x=features[0],
        y=features[1],
        hue=cluster_col,
        palette='viridis',
        s=50
    )
    plt.xlabel(features[0].upper())
    plt.ylabel(features[1].upper())
    plt.title('2D Cluster Visualization')
    plt.legend(title='Cluster')
    plt.show()

    # 3D Scatter Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        patients_df[features[0]],
        patients_df[features[1]],
        patients_df[features[2]],
        c=patients_df[cluster_col],
        cmap='viridis',
        s=50
    )
    ax.set_xlabel(features[0].upper())
    ax.set_ylabel(features[1].upper())
    ax.set_zlabel(features[2].upper())
    plt.title('3D Cluster Visualization')
    plt.colorbar(scatter, label='Cluster')
    plt.show()


def prepare_feeding_dataset(crf_df, patients_data, segments=1, interval=4 * 60, extend_min=0 * 60,
                            shift=0 * 60):
    # interval = interval * 60
    # shift = shift * 60
    sequence_size = D_0_TIME  # 1400
    new_columns = pd.DataFrame({
        'bst_mean_nz': crf_df['bst_sequence'].apply(
            lambda x: np.mean(x[:sequence_size][x[:sequence_size] != 0]) if len(
                x[:sequence_size][x[:sequence_size] != 0]) > 0 else 0),
        'insulin_sum': crf_df['insulin'].apply(lambda x: np.sum(x[:sequence_size])),
        'calorie_sum': crf_df['calorie'].apply(lambda x: np.sum(x[:sequence_size]))
    })
    crf_df = pd.concat([crf_df, new_columns], axis=1)
    gloabl_bst_mean_nz = crf_df['bst_mean_nz'].mean()
    global_insulin_sum = crf_df['insulin_sum'].mean()
    global_calorie_sum = crf_df['calorie_sum'].mean()
    bst_mean_difference = np.array(crf_df['bst_mean_nz'].tolist()) - gloabl_bst_mean_nz
    insulin_mean_difference = np.array(crf_df['insulin_sum'].tolist()) - global_insulin_sum
    calorie_mean_difference = np.array(crf_df['calorie_sum'].tolist()) - global_calorie_sum

    feeding_data = []
    for pt in crf_df.index:
        pt_data = patients_data[pt]
        ens_indices = np.nonzero(pt_data['ens'][D_0_TIME:D_1_TIME])[0] + D_0_TIME # find the time when ENS is given
        for i, ens_idx in enumerate(ens_indices): # for each feeding
            if i >= MAX_FEEDING: continue       # We only consider upto feeding_3
            start_idx = ens_idx + interval
            end_idx = ens_indices[i + 1] if i != len(ens_indices) - 1 else D_1_TIME
            if start_idx < end_idx:
                target_bst_indices = np.nonzero(pt_data['bst'][start_idx:end_idx])[0] + start_idx
                if len(target_bst_indices) > 0:
                    target_bst_idx = target_bst_indices[0]
                    target_bst = pt_data['bst'][target_bst_idx]
                    # prv_bst_indices = np.nonzero(pt_data['bst'][:ens_idx + extend_min])[0]
                    if i==0: prv_bst_indices = np.nonzero(pt_data['bst'][:ens_idx + extend_min])[0]
                    else: prv_bst_indices = np.nonzero(pt_data['bst'][ens_indices[i - 1]:ens_idx + extend_min])[0] + ens_indices[i - 1]
                    if len(prv_bst_indices)== 0: continue # no initial bst found
                    initial_bst_idx = prv_bst_indices[-1]
                    initial_bst = pt_data['bst'][initial_bst_idx]
                    if initial_bst == 0:
                        print(f"{initial_bst=}, {prv_bst_indices=} ")
                    # if target_bst_idx-initial_bst_idx > MAX_PREDICTION_DURATION: continue # total duration between initial and target BST will be with in MAX_PREDICTION_DURATION

                    # Static features collection
                    bst_slope = 0
                    if len(prv_bst_indices) >= 2:
                        pre_pre_bst_idx = prv_bst_indices[-2]
                        pre_pre_bst = pt_data['bst'][pre_pre_bst_idx]
                        bst_slope = bst_slope_calculation(initial_bst_idx, initial_bst, pre_pre_bst_idx, pre_pre_bst)
                    duration = target_bst_idx - initial_bst_idx
                    target_bst_change = target_bst - initial_bst
                    complete_insulin = pt_data['insulin'][initial_bst_idx:target_bst_idx].sum()
                    complete_insulin_iv = pt_data['insulin_iv'][initial_bst_idx:target_bst_idx].sum()
                    complete_insulin_i = pt_data['insulin_i'][initial_bst_idx:target_bst_idx].sum()
                    complete_insulin_h = pt_data['insulin_h'][initial_bst_idx:target_bst_idx].sum()
                    # complete_insulin_a = pt_data['insulin_a'][initial_bst_idx:target_bst_idx].sum()
                    complete_calorie = pt_data['calorie'][initial_bst_idx:target_bst_idx].sum()
                    complete_tpn = pt_data['tpn'][initial_bst_idx:target_bst_idx].sum()
                    complete_ens = pt_data['ens'][initial_bst_idx:target_bst_idx].sum()
                    complete_ens_d = pt_data['ens_distributed'][initial_bst_idx:target_bst_idx].sum()
                    complete_igr = complete_insulin / (complete_calorie+ 0.01)

                    intermediate_bst_indices = np.nonzero(pt_data['bst'][ ens_idx + extend_min: target_bst_idx])[0] + ens_idx + extend_min
                    intermediate_bst = pt_data['bst'][intermediate_bst_indices[-1]] if len(intermediate_bst_indices)>0 else 0

                    # Segment features collection
                    # s_insulin, s_insulin_i, s_insulin_i_iv, s_insulin_h = [], [], [], [],
                    # s_calorie, s_steroid, s_tf, s_ens_distributed, s_igr, segment_idx = [], [], [], [], [], []
                    #
                    # mean_index_i, std_dev_i, c_i = pt_data['insulin_mean'], pt_data['insulin_std'], pt_data['insulin_sensitivity']
                    # mean_index_c, std_dev_c, c_c = pt_data['calorie_mean'], pt_data['calorie_std'], pt_data['calorie_sensitivity']
                    # w_i = generate_bell_curve(16, mean_index_i, std_dev_i)
                    # w_c = generate_bell_curve(16, mean_index_c, std_dev_c)
                    #
                    # seg_start_idx = target_bst_idx - MAX_PREDICTION_DURATION
                    # extra_insulin = 0
                    # extra_calorie = 0
                    # for seg in range(segments):
                    #     seg_end_idx = seg_start_idx + SEG_WINDOW
                    #     segment_idx.append(seg_start_idx)
                    #
                    #     if seg_end_idx <= initial_bst_idx:
                    #         insulin_consumption, insulin_i, insulin_i_v, insulin_h = 0, 0, 0, 0
                    #         calorie_consumption, steroid, tf, ens_distributed, igr = 0, 0, 0, 0, 0
                    #     else:
                    #         insulin_consumption, calorie_consumption = calculate_ic_consumption(seg_start_idx,seg_end_idx,
                    #                                                                             pt_data['insulin'], w_i,c_i,
                    #                                                                             pt_data['calorie'], w_c,c_c)
                    #         if seg_start_idx <= initial_bst_idx <= seg_end_idx:
                    #             extra_portion = initial_bst_idx-seg_start_idx
                    #             extra_insulin =  (extra_portion / SEG_WINDOW) * insulin_consumption
                    #             extra_calorie =  (extra_portion / SEG_WINDOW) * calorie_consumption
                    #
                    #         insulin_i = pt_data['insulin_i'][seg_start_idx:seg_end_idx].sum()
                    #         insulin_i_v = pt_data['insulin_iv'][seg_start_idx:seg_end_idx].sum()
                    #         insulin_h = pt_data['insulin_h'][seg_start_idx:seg_end_idx].sum()
                    #         steroid = pt_data['steroid'][seg_start_idx:seg_end_idx].sum()
                    #         tf = pt_data['tpn'][seg_start_idx:seg_end_idx].sum()
                    #         ens_distributed = pt_data['ens_distributed'][seg_start_idx:seg_end_idx].sum()
                    #         igr = insulin_consumption / (calorie_consumption + 0.01)

                        # if pt in [48, 661, 51, 532, 516, 58, 599, 30, 118]:
                        #     initial_bst = initial_bst - 20
                            # insulin_consumption = 0
                            # complete_insulin = 0
                            # calorie_consumption = calorie_consumption * 100.
                            # complete_calorie = complete_calorie * 100.
                    #
                    #     s_insulin.append(insulin_consumption)
                    #     s_calorie.append(calorie_consumption)
                    #     s_insulin_i.append(insulin_i)
                    #     s_insulin_i_iv.append(insulin_i_v)
                    #     s_insulin_h.append(insulin_h)
                    #     s_steroid.append(steroid)
                    #     s_tf.append(tf)
                    #     s_ens_distributed.append(ens_distributed)
                    #     s_igr.append(igr)
                    #
                    #     seg_start_idx += SEG_WINDOW
                    #
                    # segment_idx.append(target_bst_idx)

                    single_feeding = {
                        # Patient information
                        "pt_index": pt,
                        "hospital_name": pt_data['hospital_name'],
                        "hospital_num": pt_data['hospital_num'],
                        # Feeding information
                        "feeding_start_time": pt_data['feeding_start_time'],
                        "feeding_num": int(i + 1),
                        "target_bst_idx": target_bst_idx,
                        "target_bst": target_bst,
                        "initial_bst_idx": initial_bst_idx,
                        "en_idx": ens_idx,
                        "pre_ens_time_dif": (abs(ens_idx - initial_bst_idx) // 60) +1,
                        "post_ens_time_dif": ((target_bst_idx - ens_idx) // 60) +1,
                        "initial_bst": initial_bst,
                        "initial_bst_slope": bst_slope,
                        "intermediate_bst": intermediate_bst,
                        "bst_change": target_bst_change,
                        "duration": duration,
                        # General Information
                        "age": pt_data['age'],
                        "gender": pt_data['gender'],
                        "hgt": pt_data['hgt'],
                        "bwt": pt_data['bwt'],
                        "hypo_epis": pt_data['hypo_epis'],
                        "hyper_epis": pt_data['hyper_epis'],
                        "time_in_range": pt_data['time_in_range'],
                        "cluster": pt_data['cluster'],
                        # "crrt": pt_data['crrt'],
                        # "hd": pt_data['hd'],
                        'bst_min_nz': pt_data['bst_min_nz'],
                        'bst_max': pt_data['bst_max'],
                        'bst_min_max_dif': pt_data['bst_min_max_dif'],
                        "fasting_insulin": pt_data['fasting_insulin'],
                        "fasting_calorie": pt_data['fasting_calorie'],
                        "fasting_igr": pt_data['fasting_igr'],
                        "fasting_igr_level": 1 if pt_data['fasting_igr'] > 0.3 else 0,
                        "bst_mean_difference": bst_mean_difference[pt],
                        "insulin_mean_difference": insulin_mean_difference[pt],
                        "calorie_mean_difference": calorie_mean_difference[pt],
                        # Feeding information from initial BST to final BST
                        "complete_insulin": complete_insulin,
                        # "complete_insulin_a": complete_insulin_a,
                        "complete_insulin_iv": complete_insulin_iv,
                        "complete_insulin_i": complete_insulin_i,
                        "complete_insulin_h": complete_insulin_h,
                        "complete_calorie": complete_calorie,
                        "complete_tpn": complete_tpn,
                        "complete_ens": complete_ens,
                        "complete_ens_d": complete_ens_d,
                        "complete_igr": complete_igr,
                        "combine_w_ci": complete_calorie / (0.8 * complete_insulin + 1.0),

                        # Feeding information divided into segment
                        # "segment_idx": segment_idx,
                        # "s_calorie_sum": sum(s_calorie),
                        # "s_insulin_sum": sum(s_insulin),
                        # "s_calorie": s_calorie,
                        # "s_insulin": s_insulin,
                        # "s_insulin_i": s_insulin_i,
                        # "s_insulin_i_iv": s_insulin_i_iv,
                        # "s_insulin_h": s_insulin_h,
                        # "s_steroid": s_steroid,
                        # "steroid_sum": sum(s_steroid),
                        # "s_tf": s_tf,
                        # "s_ens": s_ens_distributed,
                        # "s_igr": s_igr,
                        # "extra_insulin": extra_insulin,
                        # "extra_calorie": extra_calorie
                        # "isf": 1500 / (insulin_seq[pt, :D_0_TIME].sum() + 0.01),  # Insulin Sensitivity Factor (ISF)
                        # "cir": 500 / (calorie_seq[pt, :D_0_TIME].sum() + 0.01),  # Carbohydrate Insulin Ratio (CIR)
                        # "cluster": crf_df.loc[pt, 'cluster']
                    }
                    # if 'cluster' in crf_df.columns:
                    #     patient["cluster"] = crf_df.loc[pt, 'cluster']
                    feeding_data.append(single_feeding)

    with open(os.path.join(folder_path, 'feeding_data.pkl'), 'wb') as f:
        pickle.dump(feeding_data, f)

    return feeding_data


# def prepare_feeding_dataset(crf_df, patients_data, segments=1, interval=4 * 60, extend_min=0 * 60,
#                             shift=0 * 60):
#     # interval = interval * 60
#     # shift = shift * 60
#     sequence_size = D_0_TIME  # 1400
#     new_columns = pd.DataFrame({
#         'bst_mean_nz': crf_df['bst_sequence'].apply(
#             lambda x: np.mean(x[:sequence_size][x[:sequence_size] != 0]) if len(
#                 x[:sequence_size][x[:sequence_size] != 0]) > 0 else 0),
#         'insulin_sum': crf_df['insulin'].apply(lambda x: np.sum(x[:sequence_size])),
#         'calorie_sum': crf_df['calorie'].apply(lambda x: np.sum(x[:sequence_size]))
#     })
#     crf_df = pd.concat([crf_df, new_columns], axis=1)
#     gloabl_bst_mean_nz = crf_df['bst_mean_nz'].mean()
#     global_insulin_sum = crf_df['insulin_sum'].mean()
#     global_calorie_sum = crf_df['calorie_sum'].mean()
#     bst_mean_difference = np.array(crf_df['bst_mean_nz'].tolist()) - gloabl_bst_mean_nz
#     insulin_mean_difference = np.array(crf_df['insulin_sum'].tolist()) - global_insulin_sum
#     calorie_mean_difference = np.array(crf_df['calorie_sum'].tolist()) - global_calorie_sum
#
#     feeding_data = []
#     for pt in crf_df.index:
#         pt_data = patients_data[pt]
#         ens_indices = np.nonzero(pt_data['ens'][D_0_TIME:D_1_TIME])[0] + D_0_TIME # find the time when ENS is given
#         for i, ens_idx in enumerate(ens_indices):
#             if i >= MAX_FEEDING: continue
#             prv_bst_indices = np.nonzero(pt_data['bst'][:ens_idx + extend_min])[0]
#             if len(prv_bst_indices) == 0: continue
#             initial_bst_idx = prv_bst_indices[-1]
#             initial_bst = pt_data['bst'][initial_bst_idx]
#             start_idx = ens_idx + interval
#             end_idx = ens_indices[i + 1] if i != len(ens_indices) - 1 else D_1_TIME
#             if end_idx-initial_bst_idx > MAX_PREDICTION_DURATION:
#                 end_idx= initial_bst_idx+MAX_PREDICTION_DURATION # total duration between initial and target BST will be with in MAX_PREDICTION_DURATION
#             if start_idx < end_idx:
#                 bst_indices = np.nonzero(pt_data['bst'][start_idx:end_idx])[0] + start_idx
#                 if len(bst_indices) > 0:
#                     target_bst_idx = bst_indices[0]
#                     target_bst = pt_data['bst'][target_bst_idx]
#
#                     # Static features collection
#                     bst_slope = 0
#                     if len(prv_bst_indices) >= 2:
#                         pre_pre_bst_idx = prv_bst_indices[-2]
#                         pre_pre_bst = pt_data['bst'][pre_pre_bst_idx]
#                         bst_slope = bst_slope_calculation(initial_bst_idx, initial_bst, pre_pre_bst_idx, pre_pre_bst)
#                     duration = target_bst_idx - initial_bst_idx
#                     target_bst_change = target_bst - initial_bst
#                     complete_insulin = pt_data['insulin'][initial_bst_idx:target_bst_idx].sum()
#                     complete_calorie = pt_data['calorie'][initial_bst_idx:target_bst_idx].sum()
#                     complete_igr = complete_insulin / (complete_calorie+ 0.01)
#
#                     # Segment features collection
#                     s_insulin, s_insulin_i, s_insulin_i_iv, s_insulin_h = [], [], [], [],
#                     s_calorie, s_steroid, s_tf, s_ens_distributed, s_igr, segment_idx = [], [], [], [], [], []
#
#                     mean_index_i, std_dev_i, c_i = pt_data['insulin_mean'], pt_data['insulin_std'], pt_data['insulin_sensitivity']
#                     mean_index_c, std_dev_c, c_c = pt_data['calorie_mean'], pt_data['calorie_std'], pt_data['calorie_sensitivity']
#                     w_i = generate_bell_curve(16, mean_index_i, std_dev_i)
#                     w_c = generate_bell_curve(16, mean_index_c, std_dev_c)
#
#                     seg_start_idx = initial_bst_idx
#                     extra_insulin = 0
#                     extra_calorie = 0
#                     for seg in range(segments):
#                         seg_end_idx = seg_start_idx + SEG_WINDOW
#                         segment_idx.append(seg_end_idx)
#
#
#                         if seg_start_idx >= target_bst_idx:
#                             insulin_consumption, insulin_i, insulin_i_v, insulin_h = 0, 0, 0, 0
#                             colorie_consumption, steroid, tf, ens_distributed, igr = 0, 0, 0, 0, 0
#                         else:
#                             insulin_consumption, calorie_consumption = calculate_ic_consumption(seg_start_idx,seg_end_idx,
#                                                                                                 pt_data['insulin'], w_i,c_i,
#                                                                                                 pt_data['calorie'], w_c,c_c)
#                             if seg_start_idx < target_bst_idx <= seg_end_idx:
#                                 extra_portion = SEG_WINDOW - (target_bst_idx-seg_start_idx)
#                                 extra_insulin =  (extra_portion / SEG_WINDOW) * insulin_consumption
#                                 extra_calorie =  (extra_portion / SEG_WINDOW) * calorie_consumption
#
#
#                             insulin_i = pt_data['insulin_i'][seg_start_idx:seg_end_idx].sum()
#                             insulin_i_v = pt_data['insulin_iv'][seg_start_idx:seg_end_idx].sum()
#                             insulin_h = pt_data['insulin_h'][seg_start_idx:seg_end_idx].sum()
#                             steroid = pt_data['steroid'][seg_start_idx:seg_end_idx].sum()
#                             tf = pt_data['tpn'][seg_start_idx:seg_end_idx].sum()
#                             ens_distributed = pt_data['ens_distributed'][seg_start_idx:seg_end_idx].sum()
#                             igr = insulin_consumption / (calorie_consumption + 0.01)
#
#                         s_insulin.append(insulin_consumption)
#                         s_calorie.append(calorie_consumption)
#                         s_insulin_i.append(insulin_i)
#                         s_insulin_i_iv.append(insulin_i_v)
#                         s_insulin_h.append(insulin_h)
#                         s_steroid.append(steroid)
#                         s_tf.append(tf)
#                         s_ens_distributed.append(ens_distributed)
#                         s_igr.append(igr)
#
#                         seg_start_idx += SEG_WINDOW
#
#                     # patients_data[pt][f'feeding_{i + 1}']['seg_insulin'] = s_calorie
#                     # patients_data[pt][f'feeding_{i + 1}']['seg_calorie'] = s_insulin
#                     single_feeding = {
#                         # Patient information
#                         "pt_index": pt,
#                         "hospital_name": pt_data['hospital_name'],
#                         "hospital_num": pt_data['hospital_num'],
#                         # Feeding information
#                         "feeding_start_time": pt_data['feeding_start_time'],
#                         "feeding_num": int(i + 1),
#                         "target_bst_idx": target_bst_idx,
#                         "target_bst": target_bst,
#                         "initial_bst_idx": initial_bst_idx,
#                         "initial_bst": initial_bst,
#                         "initial_bst_slope": bst_slope,
#                         "bst_change": target_bst_change,
#                         "duration": duration,
#                         # General Information
#                         "age": pt_data['age'],
#                         "gender": pt_data['gender'],
#                         "hgt": pt_data['hgt'],
#                         "bwt": pt_data['bwt'],
#                         "hypo_epis": pt_data['hypo_epis'],
#                         "hyper_epis": pt_data['hyper_epis'],
#                         "time_in_range": pt_data['time_in_range'],
#                         "crrt": pt_data['crrt'],
#                         "hd": pt_data['hd'],
#                         "fasting_igr": pt_data['fasting_igr'],
#                         "fasting_igr_level": 1 if pt_data['fasting_igr'] > 0.3 else 0,
#                         "bst_mean_difference": bst_mean_difference[pt],
#                         "insulin_mean_difference": insulin_mean_difference[pt],
#                         "calorie_mean_difference": calorie_mean_difference[pt],
#                         # Feeding information from initial BST to final BST
#                         "complete_insulin": complete_insulin,
#                         "complete_calorie": complete_calorie,
#                         "complete_igr": complete_igr,
#                         # Feeding information divided into segment
#                         "segment_end_idx": segment_idx,
#                         "s_calorie": s_calorie,
#                         "s_insulin": s_insulin,
#                         "s_insulin_i": s_insulin_i,
#                         "s_insulin_i_iv": s_insulin_i_iv,
#                         "s_insulin_h": s_insulin_h,
#                         "s_steroid": s_steroid,
#                         "steroid_sum": sum(s_steroid),
#                         "s_tf": s_tf,
#                         "s_ens": s_ens_distributed,
#                         "s_igr": s_igr,
#                         "extra_insulin": extra_insulin,
#                         "extra_calorie": extra_calorie
#                         # "isf": 1500 / (insulin_seq[pt, :D_0_TIME].sum() + 0.01),  # Insulin Sensitivity Factor (ISF)
#                         # "cir": 500 / (calorie_seq[pt, :D_0_TIME].sum() + 0.01),  # Carbohydrate Insulin Ratio (CIR)
#                         # "cluster": crf_df.loc[pt, 'cluster']
#                     }
#                     # if 'cluster' in crf_df.columns:
#                     #     patient["cluster"] = crf_df.loc[pt, 'cluster']
#                     feeding_data.append(single_feeding)
#
#     with open(os.path.join(folder_path, 'feeding_data.pkl'), 'wb') as f:
#         pickle.dump(feeding_data, f)
#
#     return feeding_data