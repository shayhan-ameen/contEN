__author__ ='Shayhan'

import collections
import numpy as np
import matplotlib.pyplot as plt

# def group_and_plot_bin_counts(y, bin_size=20, plot_name = 'BST', display_plot_distribution=False, zero_as_extra_bin=False):
#     """
#     Group y values into categories using the bin size, then show the bin count of each category.
#
#     Parameters:
#     y (array-like): Target array with numerical values.
#     bin_size (int): Size of each bin. Default is 20.
#     """
#     # Define the bin edges with an interval of bin_size
#     min_y, max_y = min(y), max(y)
#     if zero_as_extra_bin:
#         bins = np.concatenate(([0], np.arange(min_y + 0.1, max_y + bin_size, bin_size)))
#     else:
#         bins = np.arange(min_y, max_y + bin_size, bin_size)
#
#     # Digitize y into bins
#     binned_y = np.digitize(y, bins, right=False)
#
#     if display_plot_distribution:
#
#         # Count the occurrences in each bin
#         class_counts = collections.Counter(binned_y)
#
#         # Prepare data for plotting
#         bin_labels = [f"{bins[i - 1]} - {bins[i]}" for i in range(1, len(bins))]
#         counts = [class_counts[i] for i in range(1, len(bins))]
#
#         # Plotting
#         plt.figure(figsize=(10, 6))
#         bars = plt.bar(bin_labels, counts, width=0.8, align='center')
#         plt.xlabel(f"{plot_name} Ranges (Bin size = {bin_size})")
#         plt.ylabel('Count')
#         plt.title(f"Count of Values in Each {plot_name} RANGE")
#         plt.xticks(rotation=45, ha='right')
#
#         # Adding count labels on top of each bar
#         for bar in bars:
#             yval = bar.get_height()
#             plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, str(yval), ha='center', va='bottom')
#
#         plt.tight_layout()
#         plt.show()
#     return binned_y
#
def prepare_rl_dataset(state, action, r):
    if state.shape[0] != action.shape[0] and state.shape[0] != r.shape[0]:
        raise ValueError(f"All inputs must have the same length. Currently the lentgth is {state.shape[0]}, {action.shape[0]} and {r.shape[0]}")
    dataset = []
    for i in range(state.shape[0]):
        dataset.append(((state[i]), action[i], r[i]))
    return dataset


def replace_index_with_bin_limits(Q_values_df, bst_bin_lim, calorie_bin_lim):
    """
    Replace row indices of each DataFrame in Q_values_df with values based on bst_bin_lim and calorie_bin_lim.

    Parameters:
    Q_values_df (list of DataFrames): List of DataFrames where each has row index strings representing (bst, calorie).
    bst_bin_lim (dict): Dictionary mapping bst bin numbers to (lower_limit, upper_limit) tuples.
    calorie_bin_lim (dict): Dictionary mapping calorie bin numbers to (lower_limit, upper_limit) tuples.

    Returns:
    list of DataFrames: Q_values_df with updated row indices.
    """
    # Process each DataFrame in the list Q_values_df
    updated_Q_values_df = []
    for df in Q_values_df:
        new_index = []
        for idx in df.index:
            # Check if the index is in the expected format '(x, y)'
            if idx.startswith("(") and idx.endswith(")") and "," in idx:
                # Convert the string index to a tuple
                bst, calorie = eval(idx)
                # Get the bin limits for bst and calorie
                bst_limits = bst_bin_lim.get(bst, ("Unknown", "Unknown"))
                calorie_limits = calorie_bin_lim.get(calorie, ("Unknown", "Unknown"))
                # Append the new tuple of limits
                new_index.append((bst_limits, calorie_limits))
            else:
                # Add a placeholder for incorrectly formatted index
                new_index.append(("Invalid Index", "Invalid Index"))

        # Assign the new index to the DataFrame
        df.index = new_index
        # Append the updated DataFrame to the list
        updated_Q_values_df.append(df)

    return updated_Q_values_df

def group_and_plot_bin_counts(y, bin_size=20, plot_name='BST', display_plot_distribution=False, zero_as_extra_bin=False):
    """
    Group y values into categories using the bin size, then show the bin count of each category.

    Parameters:
    y (array-like): Target array with numerical values.
    bin_size (int): Size of each bin. Default is 20.
    display_plot_distribution (bool): Whether to display a bar plot of the distribution.
    zero_as_extra_bin (bool): If True, adds zero as an extra bin lower boundary.

    Returns:
    binned_y (array): Array indicating the bin number for each value in y.
    bin_limits (dict): Dictionary mapping bin numbers to (lower_limit, upper_limit) tuples.
    """
    # Define the bin edges with an interval of bin_size
    min_y, max_y = min(y), max(y)
    if zero_as_extra_bin:
        bins = np.concatenate(([0], np.arange(min_y + 0.1, max_y + bin_size, bin_size)))
    else:
        bins = np.arange(min_y, max_y + bin_size, bin_size)

    # Digitize y into bins and adjust bin numbers to start from 0
    binned_y = np.digitize(y, bins, right=False) - 1

    # Create dictionary for bin limits, with bin numbers starting from 0
    if bin_size>1:
        bin_limits = {i: (int(bins[i]), int(bins[i + 1]-1)) for i in range(len(bins) - 1)}
    else:
        bin_limits = {i: (bins[i], bins[i + 1]) for i in range(len(bins) - 1)}

    if display_plot_distribution:
        # Count the occurrences in each bin
        class_counts = collections.Counter(binned_y)

        # Prepare data for plotting
        bin_labels = [f"{bins[i]} - {bins[i + 1]}" for i in range(len(bins) - 1)]
        counts = [class_counts[i] for i in range(len(bins) - 1)]

        # Plotting
        plt.figure(figsize=(10, 6))
        bars = plt.bar(bin_labels, counts, width=0.8, align='center')
        plt.xlabel(f"{plot_name} Ranges (Bin size = {bin_size})")
        plt.ylabel('Count')
        plt.title(f"Count of Values in Each {plot_name} RANGE")
        plt.xticks(rotation=45, ha='right')

        # Adding count labels on top of each bar
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, str(yval), ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    return binned_y, bin_limits



# def group_and_plot_bin_counts(y, bin_size=20, plot_name='BST', display_plot_distribution=False, zero_as_extra_bin=False):
#     """
#     Group y values into categories using the bin size, then show the bin count of each category.
#
#     Parameters:
#     y (array-like): Target array with numerical values.
#     bin_size (int): Size of each bin. Default is 20.
#     display_plot_distribution (bool): Whether to display a bar plot of the distribution.
#     zero_as_extra_bin (bool): If True, adds zero as an extra bin lower boundary.
#
#     Returns:
#     binned_y (array): Array indicating the bin number for each value in y.
#     bin_limits (dict): Dictionary mapping bin numbers to (lower_limit, upper_limit) tuples.
#     """
#     # Define the bin edges with an interval of bin_size
#     min_y, max_y = min(y), max(y)
#     if zero_as_extra_bin:
#         bins = np.concatenate(([0], np.arange(min_y + 0.1, max_y + bin_size, bin_size)))
#     else:
#         bins = np.arange(min_y, max_y + bin_size, bin_size)
#
#     # Digitize y into bins
#     binned_y = np.digitize(y, bins, right=False)
#
#     # Create dictionary for bin limits
#     bin_limits = {i: (int(bins[i - 1]), int(bins[i])) for i in range(1, len(bins))}
#
#     if display_plot_distribution:
#         # Count the occurrences in each bin
#         class_counts = collections.Counter(binned_y)
#
#         # Prepare data for plotting
#         bin_labels = [f"{bins[i - 1]} - {bins[i]}" for i in range(1, len(bins))]
#         counts = [class_counts[i] for i in range(1, len(bins))]
#
#         # Plotting
#         plt.figure(figsize=(10, 6))
#         bars = plt.bar(bin_labels, counts, width=0.8, align='center')
#         plt.xlabel(f"{plot_name} Ranges (Bin size = {bin_size})")
#         plt.ylabel('Count')
#         plt.title(f"Count of Values in Each {plot_name} RANGE")
#         plt.xticks(rotation=45, ha='right')
#
#         # Adding count labels on top of each bar
#         for bar in bars:
#             yval = bar.get_height()
#             plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, str(yval), ha='center', va='bottom')
#
#         plt.tight_layout()
#         plt.show()
#
#     return binned_y, bin_limits



def reverse_binning(binned_y, bins, bin_size=20, zero_as_extra_bin=False):
    """
    Reverse the binning process by mapping each binned category to a range of values.

    Parameters:
    binned_y (array-like): Binned categories to reverse.
    bins (array): Bin edges used during the binning process.
    bin_size (int): The size of each bin, which is used to reconstruct the range.
    zero_as_extra_bin (bool): If True, the first bin (0) is treated as an extra bin.

    Returns:
    reversed_values (list of tuples): List of (min, max) range tuples for each binned value.
    """
    reversed_values = []

    for bin_index in binned_y:
        if zero_as_extra_bin and bin_index == 1:
            # Handle zero bin as an extra bin
            reversed_values.append((0, bin_size))
        elif bin_index > 0 and bin_index < len(bins):
            # Get the range for the given bin index
            min_val = bins[bin_index - 1]
            max_val = bins[bin_index]
            reversed_values.append((min_val, max_val))
        else:
            # If out of range, append None or handle as appropriate
            reversed_values.append(None)

    return reversed_values




def plot_reward_function(y_values, output_figures_path= 'results'):
    # Define the function based on the given expression
    def func(y):
        return -10 * (3.5506 * (np.log(y) ** 0.8353 - 3.7932)) ** 2

    # Compute the function values for each y
    y_values = np.array(y_values)
    func_values = func(y_values)

    # Plot the function
    plt.figure(figsize=(10, 6))
    plt.plot(y_values, func_values, label=r'reward = $-10 * (3.5506 * (\log(BGL_T) ^ {0.8353} - 3.7932))^2$')
    plt.xlabel('Blood Glucose Level (mg/dL)')
    plt.ylabel('Reward')
    plt.title('Reward fucntion based on the target Blood Glucose Level')
    plt.legend()
    plt.grid(True)
    # plt.show()
    fig_path = os.path.join(output_figures_path, f"reward_function.png")
    plt.savefig(fig_path)
# # Generate y values for plotting
# y_values = np.linspace(40, 600, 100)  # Generate y values from 40 to 600
# plot_reward_function(y_values)


def print_patient_data(label, fold, patient_index, patient_data, X_test, y_test, insulins_administrated_test, predicted_insulin, feeding_info):
    print(f"-------------{label}----------------")
    print(f"{fold} - {patient_index}")
    patient = patient_data[patient_index]
    print(f"Hospital: {patient['hospital_name']}, Age: {patient['age']}, Gender: {'male' if patient['gender'] == 1 else 'female'}, Height: {patient['hgt']} cm, Weight: {patient['bwt']} kg")
    print(f"pre-ENS BGL: {X_test[0]} mg/dL, post-ENS BGL: {y_test} mg/dL,  IGR: {'high' if X_test[2] == 1 else 'low'}")
    print(f"administrated insulins: {insulins_administrated_test:.2f} units, predicted insulins: {predicted_insulin:.2f} units")
    print(f"Calorie: {feeding_info['complete_calorie']:.2f} kcal, PN: {feeding_info['complete_tpn']:.2f} kcal, ENS: {feeding_info['complete_ens_d']:.2f} kcal")
    # print(f"Feed pt id- {feeding_info['pt_index']}")
    print("-----------------------------")

    # patient = patient_data[patient_index]
    # print(f"Hospital: {patient['hospital_name']}")
    # print(f"Age: {patient['age']}")
    # print(f"Gender: {'male' if patient['gender'] == 1 else 'female'}")
    # print(f"Height: {patient['hgt']} cm")
    # print(f"Weight: {patient['bwt']} kg")
    # print(f"pre-ENS BGL: {X_test[0]} mg/dL")
    # print(f"Calorie: {X_test[0]}  kca")
    # print(f"IGR: {'high' if X_test[2]==1 else 'low'}")
    # print(f"post-ENS BGL: {y_test} mg/dL")
    # print(f"administrated insulins: {insulins_administrated_test} unites")
    # print(f"predicted insulins: {predicted_insulin} unites")
    print("-----------------------------")

def check_and_print_condition(label, fold, flt_pts_test, patients_data, X_test, y_test, insulins_administrated_test, predicted_insulin, condition):
    for i in range(len(X_test)):
        if condition(X_test[i], y_test[i], insulins_administrated_test[i], predicted_insulin[i]):
            print_patient_data(label, fold, flt_pts_test[i], patients_data, X_test[i], y_test[i], insulins_administrated_test[i], predicted_insulin[i])
            # print(f"State: {X_test[i]}, Target: {y_test[i]}, Action: {insulins_administrated_test[i]}, Prediction: {predicted_insulin[i]}")

