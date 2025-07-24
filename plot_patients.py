__author__ = 'Shayhan'

from scipy.stats import spearmanr
from sympy.printing.pretty.pretty_symbology import line_width

from constants import *
from utils import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import Rectangle
import os

def plot_reward_function(y_values, output_figures_path='results'):
    # Define the reward function
    def func(y):
        return -10 * (3.5506 * (np.log(y) ** 0.8353 - 3.7932)) ** 2

    # Compute the function values for each y
    y_values = np.array(y_values)
    func_values = func(y_values)

    # Set plot style
    sns.set_style("whitegrid", {"grid.color": ".6", "grid.linestyle": ":"})

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(y_values, func_values, label='Reward Function', linewidth=2, color='blue')
    plt.xlabel('Blood Glucose Level (mg/dL)')
    plt.ylabel('Reward')

    # Add vertical lines for reference points
    # plt.axvline(80, color='gray', linestyle='dashed', linewidth=1)  # Line at x = 80
    # plt.axvline(180, color='gray', linestyle='dashed', linewidth=1)  # Line at x = 180

    # Fill green region for x between 80 and 180
    plt.axvspan(80, 180, color='green', alpha=0.2, label='Normal Range (80-180 mg/dL)')

    # Fill red region
    plt.axvspan(30, 80, color='red', alpha=0.3, label='Below Normal (<80 mg/dL)')
    plt.axvspan(180, 600, color='red', alpha=0.1, label='Above Normal (>180 mg/dL)')

    # Set x-axis limits to remove white space
    plt.xlim(30, 600)
    xticks = plt.xticks()[0]  # Get current x-ticks
    plt.xticks(np.arange(30, 600, 50))

    # Add grid and legend
    plt.grid(True)
    plt.legend()

    # Save the plot to the specified path
    os.makedirs(output_figures_path, exist_ok=True)  # Ensure the output directory exists
    fig_path = os.path.join(output_figures_path, "reward_function.eps")
    plt.savefig(fig_path)
    plt.close()  # Close the plot to free memory

# Generate y values for plotting and call the function
# y_values = np.linspace(30, 600, 50)  # Generate smooth y values from 30 to 600
# plot_reward_function(y_values)

def plot_bst_vs_ins(bst_difs, ins_difs, plot_name="", category=None):
    # Calculate the correlation coefficient
    correlation = np.corrcoef(bst_difs, ins_difs)[0, 1]
    print(f"Correlation ({plot_name}) = {correlation:.4f}")

    plt.figure(figsize=(8, 6))

    # Define the Estimation line points
    # x_points = [-60, -50, 0, 50, 100, 150, 200, 250, 300, 350, 400, 450]
    # y_points = [-.67, -.67, 0, .67, 1.33, 2, 2.67, 3.33, 4, 4.67, 5.33, 6.0]

    # x_points = [-60, -50, 0, 50, 100, 150, 200, 250, 300, 350, 400, 450]
    # y_points = [0, 0, 0, .67, 1.33, 2, 2.67, 3.33, 4, 4.67, 5.33, 6.0]

    x_points = [-60, -50, 0, 50, 100, 150, 200, 250, 300, 350, 400, 450]
    y_points = [-.8, -.67, 0, .67, 1.33, 2, 2.67, 3.33, 4, 4.67, 5.33, 6.0]

    # Classify points based on proximity to the Estimation line
    estimation_line = np.interp(bst_difs, x_points, y_points)  # Estimate y-values based on bst_difs x-values
    correct = np.abs(ins_difs - estimation_line) <= 5.5
    # correct = ((ins_difs - estimation_line)<=2) & ((ins_difs - estimation_line) >= -1)
    incorrect = ~correct

    # Update category to reflect correct/incorrect classification
    if category is None:
        category = np.zeros_like(bst_difs)  # Default to incorrect (0)
    category[correct] = 1  # Set correct points to 1

    # Define colors based on category: 1 for correct (green), 0 for incorrect (orange)
    colors = ['green' if c == 1 else 'red' for c in category]

    # Scatter plot with colors based on category
    plt.scatter(bst_difs, ins_difs, alpha=0.7, c=colors)

    plot_full_name = plot_name
    if plot_name == "in_range":
        plot_full_name = "IN DESIRED RANGE"
    elif plot_name == "out_range":
            plot_full_name = "OUT OF DESIRED RANGE"
    else:
        plot_full_name = "ALL"

    # Labels and title
    plt.xlabel(r"$BST_{\text{TARGET}} - 130$")
    plt.ylabel(r"$\text{INS}_{\text{Predicted}}$ - $\text{INS}_{\text{Administered}}$")
    plt.title(f"Scatter Plot of BST Differences vs. INSULIN Differences ({plot_full_name.upper()})")
    plt.grid(True)

    # Add lines at (0,0) in both x and y directions
    plt.axhline(0, color='black', linestyle='solid', linewidth=1)  # Horizontal line at y=0
    plt.axvline(0, color='black', linestyle='solid', linewidth=1)  # Vertical line at x=0

    # Plot the Estimation line
    plt.plot(x_points, y_points, label="Estimation", color="darkorange", linestyle='--', linewidth=3)

    # Add the correlation coefficient as text on the plot
    # plt.text(0.05, 0.95, f"Total: {len(bst_difs)} | Corr.: {correlation:.4f}", transform=plt.gca().transAxes,
    #          ha='left', va='top', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    # plt.text(0.05, 0.95,
    #          f"Corr.: {correlation:.4f} | Total: {len(bst_difs)} | Correct: {np.sum(correct)} | Incorrect: {np.sum(incorrect)}  | Acc: {np.sum(correct)/len(bst_difs):.2f}",
    #          transform=plt.gca().transAxes,
    #          ha='left', va='top', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    plt.text(0.05, 0.95,
             f"Acc: {(np.sum(correct) / len(bst_difs))*100:.2f}  | Correct: {np.sum(correct)} | Incorrect: {np.sum(incorrect)}  | Total: {len(bst_difs)} | Corr.: {correlation:.4f}",
             transform=plt.gca().transAxes,
             ha='left', va='top', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    # Custom legend for categories
    if category is not None:
        plt.scatter([], [], color='green', label='Correct prediction')
        plt.scatter([], [], color='red', label='Incorrect prediction')
        plt.legend(loc='lower right')

    # Save the plot to the current directory with a default filename
    fig_path = os.path.join(output_figures_path, f"BST_VS_INS_{plot_name}.png")
    plt.savefig(fig_path)
    plt.close()


def plot_bst_vs_ins_quadrant(bst_difs, ins_difs, plot_name="", epochs=25, category=None):
    x_ori = 130
    y_ori = 0
    bst_difs = np.array(bst_difs)
    ins_difs = np.array(ins_difs)
    plot_full_name = "ALL"
    # sns.set_style("whitegrid", {"grid.color": ".6", "grid.linestyle": ""})
    # sns.set_style("whitegrid")
    sns.set_style("whitegrid", {
        "grid.color": "0.7",  # Grid color (black)
        "grid.linestyle": "-",  # Solid line
    })

    plt.figure(figsize=(8, 6))
    plt.ylim(-30, 20)  # Set y-axis range from -30 to +20

    # Define colors based on category
    colors = np.where((bst_difs >= x_ori) & (ins_difs >= y_ori), 'forestgreen',  # Q1
                      np.where((bst_difs < x_ori) & (ins_difs > y_ori), 'orangered',  # Q2
                               np.where((bst_difs <= x_ori) & (ins_difs <= y_ori), '#1f77b4',  # Q3
                                        'darkorange')))  # Q4

    # Scatter plot with colors based on quadrant
    plt.scatter(bst_difs, ins_difs, alpha=0.7, c=colors)
    # Add custom x-ticks
    plt.xticks(np.arange(30, 600, 50))

    # Draw rectangles for quadrants
    # Q1: Top-right
    plt.gca().add_patch(plt.Rectangle((x_ori, y_ori), 500, 500, color='forestgreen', alpha=0.1, label="Q1"))
    # Q2: Top-left
    plt.gca().add_patch(plt.Rectangle((0, y_ori), x_ori, 500, color='orangered', alpha=0.1, label="Q2"))
    # Q3: Bottom-left
    plt.gca().add_patch(plt.Rectangle((0, -500), x_ori, 500, color='#1f77b4', alpha=0.1, label="Q3"))
    # Q4: Bottom-right
    plt.gca().add_patch(plt.Rectangle((x_ori, -500), 500, 500, color='darkorange', alpha=0.1, label="Q4"))

    # Labels and title
    plt.xlabel("Post-EN BGL Level")
    # plt.xlabel(r"$BGL_{\text{TARGET}} - BGL_{\text{DESIRED}} (130)$")
    # plt.ylabel(r"$\text{INS}_{\text{Predicted}}$ - $\text{INS}_{\text{Administered}}$")
    plt.ylabel("INSULIN (Predicted - Administered)")
    # plt.title(f"Scatter Plot of Insulin Differences Across Various BGL Levels")

    plt.grid(True)

    # Add lines at (0,0) in both x and y directions
    plt.axhline(0, color='black', linestyle='solid', linewidth=3)  # Horizontal line at y=0
    plt.axvline(x_ori, color='black', linestyle='solid', linewidth=3)  # Vertical line at x=0

    # Count the points in each quadrant
    q1 = np.sum((bst_difs >= x_ori) & (ins_difs >= y_ori))
    q2 = np.sum((bst_difs < x_ori) & (ins_difs > y_ori))
    q3 = np.sum((bst_difs <= x_ori) & (ins_difs <= y_ori))
    q4 = np.sum((bst_difs > x_ori) & (ins_difs < y_ori))

    acc = (q1 + q3) / len(bst_difs) * 100
    score = (q1 + q3 - q2 - q4) / len(bst_difs)

    print(f"Accuracy: {acc:.1f}% | A: {q1} | B: {q2} | C: {q3} | D: {q4}")
    print(f"Accuracy: {acc:.1f}%")

    displacement = 10  # 10
    #
    #  # Add labels for quadrants with black outlines
    plt.text(
        x_ori + 400, y_ori + displacement, "A", fontsize=16, color='forestgreen', fontweight='bold',
        path_effects=[path_effects.withStroke(linewidth=1, foreground="black")]
    )  # Q1
    plt.text(
        x_ori - 50, y_ori + displacement, "B", fontsize=16, color='orangered', fontweight='bold', ha='right',
        path_effects=[path_effects.withStroke(linewidth=1, foreground="black")]
    )  # Q2
    plt.text(
        x_ori - 50, y_ori - displacement, "C", fontsize=16, color='#1f77b4', fontweight='bold', ha='right',
        path_effects=[path_effects.withStroke(linewidth=1, foreground="black")]
    )  # Q3
    plt.text(
        x_ori + 400, y_ori - displacement, "D", fontsize=16, color='darkorange', fontweight='bold',
        path_effects=[path_effects.withStroke(linewidth=1, foreground="black")]
    )  # Q4

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True)  # Ensure all spines are visible
        spine.set_linewidth(0.5)  # Set border width
        spine.set_color('black')

        plt.text(0.95, 0.05, f"Accuracy: {acc:.1f}% | A: {q1} | B: {q2} | C: {q3} | D: {q4}",
                 transform=plt.gca().transAxes,
                 ha='right', va='bottom', fontsize=14, bbox=dict(facecolor='white', alpha=0.5))

        # Save the plot to the specified path
    # fig_path = os.path.join(output_figures_path, f"BST_VS_INS_{plot_name}_quadrant.png")
    fig_path = os.path.join(output_figures_path, f"BST_VS_INS_{plot_name}_quadrant_bci__{epochs}.png")
    # plt.savefig(fig_path)
    plt.savefig(fig_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return acc
# with open(os.path.join(folder_path, 'BST_DIFS.pkl'), 'rb') as f: BST_DIFS = pickle.load(f)
# with open(os.path.join(folder_path, 'INS_DIFS.pkl'), 'rb') as f: INS_DIFS = pickle.load(f)
# plot_bst_vs_ins_quadrant(BST_DIFS, INS_DIFS, "all")

# def plot_bst_vs_ins_quadrant(bst_difs, ins_difs, plot_name="", category=None):
#     # Convert bst_difs and ins_difs to numpy arrays to enable element-wise operations
#     x_ori = 130
#     y_ori = 0
#     bst_difs = np.array(bst_difs)
#     ins_difs = np.array(ins_difs)
#
#     # Calculate the correlation coefficient
#     correlation = np.corrcoef(bst_difs, ins_difs)[0, 1]
#     print(f"Correlation ({plot_name}) = {correlation:.4f}")
#
#     # Calculate Spearman correlation
#     # correlation, p_value = spearmanr(bst_difs, ins_difs)
#     # print("Spearman Correlation Coefficient:", correlation)
#     # print("P-value:", p_value)
#
#     plt.figure(figsize=(8, 6))
#
#     # Define colors based on category
#     if category is not None:
#         colors = ['darkorange' if c == 0 else '#1f77b4' for c in category]
#         color_map = {'DUMC': 'darkorange', 'KNUH': '#1f77b4', 'KHNMC': 'red', 'KHNMC_V2': 'red'}
#         colors = [color_map.get(h, 'black') for h in category]
#
#     else:
#         # colors = '#1f77b4'  # Default color if category is not provided
#         # Define colors based on quadrants
#         colors = np.where((bst_difs >= x_ori) & (ins_difs >= y_ori), 'forestgreen',  # Q1
#                           np.where((bst_difs < x_ori) & (ins_difs > y_ori), 'orangered',  # Q2
#                                    np.where((bst_difs <= x_ori) & (ins_difs <= y_ori), '#1f77b4',  # Q3
#                                             'darkorange')))  # Q4
#
#     # Scatter plot with colors based on quadrant
#     plt.scatter(bst_difs, ins_difs, alpha=0.7, c=colors)
#     # Add custom x-ticks
#     xticks = plt.xticks()[0]  # Get current x-ticks
#     # plt.xticks(list(xticks) + [80, 130, 180])  # Add -50 and 50 to the existing ticks
#     plt.xticks(np.arange(30, 600, 50))
#
#     # Draw rectangles for quadrants
#     # Q1: Top-right
#     plt.gca().add_patch(plt.Rectangle((x_ori, y_ori), 500, 500, color='forestgreen', alpha=0.1, label="Q1"))
#     # Q2: Top-left
#     plt.gca().add_patch(plt.Rectangle((0, y_ori), x_ori, 500, color='orangered', alpha=0.1, label="Q2"))
#     # Q3: Bottom-left
#     plt.gca().add_patch(plt.Rectangle((0, -500), x_ori, 500, color='#1f77b4', alpha=0.1, label="Q3"))
#     # Q4: Bottom-right
#     plt.gca().add_patch(plt.Rectangle((x_ori, -500), 500, 500, color='darkorange', alpha=0.1, label="Q4"))
#
#     #  # Add labels for quadrants
#     # plt.text(x_ori + 400, y_ori + 15, "A", fontsize=16, color='forestgreen', fontweight='bold')  # Q1
#     # plt.text(x_ori - 50, y_ori + 15, "B", fontsize=16, color='orangered', fontweight='bold', ha='right')  # Q2
#     # plt.text(x_ori - 50, y_ori - 15, "C", fontsize=16, color='#1f77b4', fontweight='bold', ha='right')  # Q3
#     # plt.text(x_ori + 400, y_ori - 15, "D", fontsize=16, color='darkorange', fontweight='bold')  # Q4
#
#     plot_full_name = plot_name
#     if plot_name == "in_range":
#         plot_full_name = "IN DESIRED RANGE"
#     elif plot_name == "out_range":
#         plot_full_name = "OUT OF DESIRED RANGE"
#     else:
#         plot_full_name = "ALL"
#
#     # Labels and title
#     plt.xlabel("Post-EN BGL Level")
#     # plt.xlabel(r"$BGL_{\text{TARGET}} - BGL_{\text{DESIRED}} (130)$")
#     plt.ylabel(r"$\text{INS}_{\text{Predicted}}$ - $\text{INS}_{\text{Administered}}$")
#     # plt.title(f"Scatter Plot of Insulin Differences Across Various BGL Levels")
#     if category is not None: plt.legend(handles=[plt.Line2D([0], [0], color=color_map[key], label=key, lw=4) for key in color_map], loc="upper right")
#     plt.grid(True)
#
#     # Add lines at (0,0) in both x and y directions
#     plt.axhline(0, color='black', linestyle='solid', linewidth=3)  # Horizontal line at y=0
#     plt.axvline(x_ori, color='black', linestyle='solid', linewidth=3)  # Vertical line at x=0
#
#     # Count the points in each quadrant
#     q1 = np.sum((bst_difs >= x_ori) & (ins_difs >= y_ori))
#     q2 = np.sum((bst_difs < x_ori) & (ins_difs > y_ori))
#     q3 = np.sum((bst_difs <= x_ori) & (ins_difs <= y_ori))
#     q4 = np.sum((bst_difs > x_ori) & (ins_difs < y_ori))
#
#     acc = (q1 + q3) / len(bst_difs)*100
#     score = (q1 + q3 - q2 - q4)/ len(bst_difs)
#
#     # Add vertical lines at x = -50 and x = 50
#     plt.axvline(80, color='gray', linestyle='dashed', linewidth=1)  # Vertical line at x = -50
#     plt.axvline(180, color='gray', linestyle='dashed', linewidth=1)  # Vertical line at x = 50
#
#
#     # Add the correlation coefficient and quadrant counts as text on the plot
#     # plt.text(0.95, 0.15, f"Total: {len(bst_difs)} |  Corr.: {correlation:.2f}", transform=plt.gca().transAxes,
#     #          ha='right', va='bottom', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
#     # plt.text(0.95, 0.05, f"Accuracy: {acc:.1f}% | A: {q1} | B: {q2} | C: {q3} | D: {q4}",
#     #          transform=plt.gca().transAxes,
#     #          ha='right', va='bottom', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
#     print(f"Accuracy: {acc:.1f}% | A: {q1} | B: {q2} | C: {q3} | D: {q4}")
#     print(f"Accuracy: {acc:.1f}%")
#     # Custom legend for quadrants
#     # plt.scatter([], [], color='green', label='Q1 (BST>0, INS>0)')
#     # plt.scatter([], [], color='blue', label='Q2 (BST<0, INS>0)')
#     # plt.scatter([], [], color='purple', label='Q3 (BST<0, INS<0)')
#     # plt.scatter([], [], color='red', label='Q4 (BST>0, INS<0)')
#     # plt.legend(loc='lower left')
#
#     displacement = 1 # 10
#
#      # Add labels for quadrants with black outlines
#     plt.text(
#         x_ori + 400, y_ori + displacement, "A", fontsize=16, color='forestgreen', fontweight='bold',
#         path_effects=[path_effects.withStroke(linewidth=1, foreground="black")]
#     )  # Q1
#     plt.text(
#         x_ori - 50, y_ori + displacement, "B", fontsize=16, color='orangered', fontweight='bold', ha='right',
#         path_effects=[path_effects.withStroke(linewidth=1, foreground="black")]
#     )  # Q2
#     plt.text(
#         x_ori - 50, y_ori - displacement, "C", fontsize=16, color='#1f77b4', fontweight='bold', ha='right',
#         path_effects=[path_effects.withStroke(linewidth=1, foreground="black")]
#     )  # Q3
#     plt.text(
#         x_ori + 400, y_ori - displacement, "D", fontsize=16, color='darkorange', fontweight='bold',
#         path_effects=[path_effects.withStroke(linewidth=1, foreground="black")]
#     )  # Q4
#
#     # Save the plot to the specified path
#     # fig_path = os.path.join(output_figures_path, f"BST_VS_INS_{plot_name}_quadrant.png")
#     fig_path = os.path.join(output_figures_path, f"BST_VS_INS_{plot_name}_quadrant_bci.png")
#     plt.savefig(fig_path)
#     plt.close()
#     return acc









# def plot_bst_vs_ins(bst_difs, ins_difs, plot_name="", category=None):
#     # Calculate the correlation coefficient
#     correlation = np.corrcoef(bst_difs, ins_difs)[0, 1]
#     print(f"Correlation ({plot_name}) = {correlation:.4f}")
#
#     plt.figure(figsize=(8, 6))
#
#     # Define colors based on category
#     if category is not None:
#         colors = ['darkorange' if c == 0 else '#1f77b4' for c in category]
#     else:
#         colors = '#1f77b4'  # Default color if category is not provided
#
#     # Scatter plot with colors based on category
#     plt.scatter(bst_difs, ins_difs, alpha=0.7, c=colors)
#
#     plot_full_name = plot_name
#     if plot_name == "in_range":
#         plot_full_name = "IN DESIRED RANGE"
#     elif plot_name == "out_range":
#             plot_full_name = "OUT OF DESIRED RANGE"
#     else:
#         plot_full_name = "ALL"
#
#     # Labels and title
#     plt.xlabel(r"$BST_{\text{TARGET}} - 130$")
#     plt.ylabel(r"$\text{INS}_{\text{Predicted}}$ - $\text{INS}_{\text{Administered}}$")
#     plt.title(f"Scatter Plot of BST Differences vs. INSULIN Differences ({plot_full_name.upper()})")
#     plt.grid(True)
#
#     # Add lines at (0,0) in both x and y directions
#     plt.axhline(0, color='black', linestyle='solid', linewidth=1)  # Horizontal line at y=0
#     plt.axvline(0, color='black', linestyle='solid', linewidth=1)  # Vertical line at x=0
#
#     # Add the correlation coefficient as text on the plot
#     plt.text(0.05, 0.95, f"Total: {len(bst_difs)} | Corr.: {correlation:.4f}", transform=plt.gca().transAxes,
#              ha='left', va='top', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
#
#     # Custom legend for categories
#     if category is not None:
#         plt.scatter([], [], color='#1f77b4', label='Correct')
#         plt.scatter([], [], color='darkorange', label='Incorrect')
#         plt.legend(loc='lower left')
#
#
#     # Save the plot to the current directory with a default filename
#     fig_path = os.path.join(output_figures_path,f"BST_VS_INS_{plot_name}.png")
#     plt.savefig(fig_path)
#     plt.close()




def plot_patients_data(folder_path, patients_list, patients_data, fold_num, errors=[], predictions_info=[], feeding_num=1, D_0_TIME=1440, D_1_TIME=2880, PATIENT_TYPE=None, messages=None):

    # Parallel(n_jobs=-1)(delayed(plot_single_patient_data)(folder_path,  pt, patients_data[pt], fold_num,
    #                                                       error=errors[i] if len(errors)>0 else None,
    #                                                       prediction_info=predictions_info[i] if len(predictions_info) > 0 else None,
    #                                                       feeding_num = feeding_num[i] if isinstance(feeding_num, list) else feeding_num,
    #                                                       D_1_TIME=D_1_TIME, D_0_TIME=D_0_TIME, PATIENT_TYPE=PATIENT_TYPE) for i, pt in enumerate(patients_list))

    for i, pt in enumerate(patients_list):
        # if i!=8: continue
        # if i!=99: continue
        # if i not in [1, 5, 177, 221, 139]: continue
        simple_plot_single_patient_data(folder_path,  pt, patients_data[pt], fold_num,
                                  error=errors[i] if len(errors)>0 else None,
                                  prediction_info=predictions_info[i] if len(predictions_info) > 0 else None,
                                  feeding_num = feeding_num[i] if isinstance(feeding_num, list) else feeding_num,
                                  D_1_TIME=D_1_TIME, D_0_TIME=D_0_TIME, PATIENT_TYPE=PATIENT_TYPE)

                                 # , message=messages[i]  if len(messages)>0 else None)

    # Plot histogram of errors
    # plt.figure()
    # errors = np.array(errors)
    # n, bins, patches = plt.hist(errors, bins=20, color='skyblue', edgecolor='black')
    # plt.xlabel('Error Value (RMSE)')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Errors (Test Dataset)')
    # plt.xticks(np.arange(0, errors.max(), step=int(errors.max() / 20.)), rotation=45, ha='right', rotation_mode='anchor')
    # # Adding the count on top of each bin
    # for count, patch in zip(n, patches):
    #     plt.text(patch.get_x() + patch.get_width() / 2, count, int(count), ha='center', va='bottom')
    # plt.tight_layout()
    # # Save histogram
    # hist_path = os.path.join(folder_path, f"errors/fold_errors/{PATIENT_TYPE}/", f"fold_{fold}_RMSE_Hist.png")
    # plt.savefig(hist_path)
    # plt.close()


def plot_single_patient_data(folder_path,  pt, pt_data, fold_num, error=None, prediction_info=None, feeding_num=1, D_0_TIME=1440, D_1_TIME=2880, PATIENT_TYPE=None, message=None):
    # print(f"{feeding_num=} | {error=}")
    # if error is None: return
    error = 0


    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    # gray_color = '#f0f0f0'
    gray_color = '#ffffff'
    color = 'darkorange'

    if pt != pt_data["index"]:
        print("++++++++++++++++++++++++++Problem++++++++++++++++++++++++++++")

    actual_bst = pt_data['bst'][:D_1_TIME]
    mask = actual_bst != 0
    actual_bst = actual_bst[mask]
    # for i, m in enumerate(mask):
    #     if m == True:
    #         print(i)
    # feeding_start, feeding_end = 0, 0
    feeding_start, feeding_end = 1414, 1981


    fig = plt.figure(figsize=(12, 14))
    fontsize = 16
    ax = {}  # Dictionary to store axes for each subplot
    if message is None:
        gs = gridspec.GridSpec(5, 1, height_ratios=[2, 1, 1, 1, 1])  # Define the ratio of heights for each row
        ax1 = plt.subplot(gs[0])
    else:
        gs = gridspec.GridSpec(6, 1, height_ratios=[1, 2, 1, 1, 1, 1])  # Define the ratio of heights for each row

        # Display the Q-Value table
        # Reset the index to make it a column, so we can display it in the table
        message.reset_index(inplace=True)
        message.columns = ['State (BGL, Calorie)'] + list(message.columns[1:])  # Rename columns to include the new "State" column
        # message.columns = ['State'] + list(message.columns[1:])  # Rename columns to include the new "State" column
        message.replace(to_replace=r'nan \(nan\)', value='', regex=True, inplace=True)  # Replace "nan (nan)" with an empty string
        message.replace(to_replace=np.nan, value='', inplace=True)
        ax0 = plt.subplot(gs[0])
        ax0.axis('off')  # Hide the plot axes
        table = ax0.table(cellText=message.values, colLabels=message.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)  # Disable automatic font scaling
        table.set_fontsize(11)  # Set desired font size for the table
        # Adjust the cell width and height
        for (row, col), cell in table.get_celld().items():
            if col == 0:
                cell.set_width(0.2)  # Set a wider width for the first column
            else:
                cell.set_width(0.08)  # Set the standard width for other columns
            cell.set_height(0.3)  # Set desired height for all cells
        # ax0.set_title(" Q-Values Table", fontsize=12)

        # Main plot: Plot actual BST and predictions (if available)
        ax1 = plt.subplot(gs[1])
    ax1.plot(np.where(mask)[0], actual_bst, label='Actual BST', marker='x') #color='orange'

    # if feeding_start != 0:
    #     indices = []
    #     temp_bst = []
    #     for m, b in zip(np.where(mask)[0], actual_bst):
    #         if feeding_start<=m<=feeding_end:
    #             indices.append(m)
    #             temp_bst.append(b)
    #     ax1.plot(indices, temp_bst, label='Actual BST', marker='x', color='orange')



    segment_info = None
    # Plot predictions if available
    if prediction_info:
        # ax1.plot(pt_data[f"feeding_{feeding_num}"]['target_bst_idx'], prediction_info, label='Predicted BST', linestyle='--', marker='^',
        #          color='green')
        ax1.plot(pt_data[f"feeding_{feeding_num}"]['initial_bst_idx'], pt_data[f"feeding_{feeding_num}"]['initial_bst'], label='Initial BST', linestyle='None',
                 marker='o', color='black')
        ax1.plot(pt_data[f"feeding_{feeding_num}"]['target_bst_idx'], pt_data[f"feeding_{feeding_num}"]['target_bst'], label='Target BST', linestyle='None',
                 marker='D', color='red')

        # seg_insulin = pt_data[f"feeding_{feeding_num}"]['seg_insulin']
        # seg_calorie = pt_data[f"feeding_{feeding_num}"]['seg_calorie']
        complete_insulin = pt_data[f"feeding_{feeding_num}"]['complete_insulin']
        complete_calorie = pt_data[f"feeding_{feeding_num}"]['complete_calorie']
        I_BST = pt_data[f'feeding_{feeding_num}']['initial_bst']
        I_BST = pt_data[f'feeding_{feeding_num}']['initial_bst']
        T_BST = pt_data[f"feeding_{feeding_num}"]['target_bst']
        total_duration = pt_data[f"feeding_{feeding_num}"]['duration']
        segment_info = (f"\n (Initial BST: {I_BST} | Calorie: {complete_calorie:.1f}) Target BGL: {T_BST}" +
                        f"\n Hourly Insulin: Predicted= {prediction_info:.2f} |  Administered= {(complete_insulin*60.)/total_duration:.2f} ")
        # segment_info=(f"\nInsulin # Complete:{complete_insulin:.2f} | Segment:{[f'{value:.2f}' for value in seg_insulin]}"+
        #               f"\nCalorie # Complete:{complete_calorie:.2f} | Segment:{[f'{value:.2f}' for value in seg_calorie]}")

    # Customize main plot

    # print(message)

    fig.suptitle(f'Hospital: {pt_data["hospital_name"]} | Patient No: {pt_data["pt_num"]} '
              f'\n  Feeding Start Time: {pt_data["feeding_start_time"]} '
              f'\n Index = {pt} | Feeding Num: {feeding_num} | Error: {error:.4f}'
              f'{segment_info if segment_info else ""}',
              # fontsize=8)
              fontsize=fontsize + 6)

    # ax1.set_title(f'Hospital: {pt_data["hospital_name"]} | Patient No: {pt_data["pt_num"]} '
    #               f'\n  Feeding Start Time: {pt_data["feeding_start_time"]} '
    #               f'\n Index = {pt} | Feeding Num: {feeding_num} | Error: {error:.4f}'
    #               f'{segment_info if segment_info else ""}',
    #               # fontsize=8)
    #               fontsize=fontsize + 6)
    # Custom subtitle (adjust coordinates as needed)

    ax1.axvline(x=D_0_TIME, color='red', linestyle='--', label=f'Line at {D_0_TIME // 60} hours')
    # for i, segment_line in enumerate(pt_data[f"feeding_{feeding_num}"]['segment_idx']):
    #     if i == 0:
    #         ax1.axvline(x=segment_line, color='gray', linestyle=(0, (1, 10)), label='Segments')  # 'loosely dotted',     (0, (1, 10)))
    #     else:
    #         ax1.axvline(x=segment_line, color='gray', linestyle=(0, (1, 10)))  # No label for subsequent lines

    step = 2 * 60  # 2 hours
    ax1.set_xticks(np.arange(0, D_1_TIME, step=step))
    xlabels = [str(i // 60 + 1) for i in range(0, D_1_TIME, step)]
    ax1.set_xticklabels(xlabels, rotation=45)
    ax1.set_xlabel('Time Steps (Hours)', fontsize=fontsize)
    ax1.set_ylabel('BGL', fontsize=fontsize)
    # legend = ax1.legend(loc='lower left', fontsize=fontsize)
    # legend.get_frame().set_facecolor(gray_color)
    ax1.grid(True)

    # Create and plot subplots for other data (excluding ignored chart data)

    temporal_features = {
            'Total Calories': pt_data['calorie'],
            # 'Total Calories': pt_data['calorie_a'],
            'Insulin': pt_data['insulin'],
            # 'Insulin': pt_data['insulin_a'],
            'Steroid': pt_data['steroid'],

            'TPN / Fluid': pt_data['tpn'],
            'Medication': pt_data['medication'],
            'Insulin-I': pt_data['insulin_i'],
            'Insulin-H': pt_data['insulin_h'],
            'Insulin-IV': pt_data['insulin_iv'],
            'Insulin-A': pt_data['insulin_a']
        }
    ignore_temporal_features = ['TPN / Fluid', 'Insulin-I', 'Insulin-H', 'Insulin-IV', 'Insulin-A']
    j = 2 if message is not None else 1
    for key in temporal_features.keys():
        # print(f"{j=}")
        if key in ignore_temporal_features: continue
        ax[key] = plt.subplot(gs[j], sharex=ax1)
        # if key == 'Insulin':
        # #     # ax[key].plot(all_features['Insulin-H'][pt, :D_1_TIME], label='Insulin-H', linestyle='--', color=color)
        # #     # ax[key].plot(all_features['Insulin-I'][pt, :D_1_TIME], label='Insulin-I', linestyle='--', color='red')
        # #     # ax[key].plot(temporal_features['Insulin-IV'][:D_1_TIME], label='Insulin-IV', linestyle='dotted', color=color)
        #     ax[key].plot(temporal_features['Insulin-A'][:D_1_TIME], label='Insulin-A', linestyle='dotted', color=color)

        ax[key].plot(temporal_features[key][ :D_1_TIME], label=key)
        if feeding_start !=0:
            if key!='Total Calories':
                ax[key].plot( np.arange(feeding_start, feeding_end),temporal_features[key][feeding_start:feeding_end], color='darkorange', linewidth=3)
            else:
                ax[key].plot(np.arange(feeding_start, feeding_end), temporal_features[key][feeding_start:feeding_end], color='fuchsia', linewidth=3)
        ax[key].axvline(x=D_0_TIME, color='red', linestyle='--')
        ax[key].set_xlabel('Time Steps (Hours)', fontsize=fontsize)
        ax[key].set_ylabel(key, fontsize=fontsize)
        # legend = ax[key].legend(loc='lower left', fontsize=fontsize)
        # legend.get_frame().set_facecolor(gray_color)
        ax[key].grid(True)
        j += 1

    # Plot TPN/Fluid on twin axis of Total Calories
    # key = 'TPN / Fluid'
    # ax[key] = ax['Total Calories'].twinx()
    # ax[key].plot(temporal_features[key][:D_1_TIME], color=color, label=key)
    # legend = ax[key].legend(loc='lower right', fontsize=fontsize)
    # legend.get_frame().set_facecolor(gray_color)
    # plt.tight_layout(pad=2)

    # Final adjustments and save figure
    plt.tight_layout(pad=2)
    if PATIENT_TYPE:
        # fig_path = os.path.join(folder_path, f"errors/all_patients/", f"{PATIENT_TYPE}_error_{int(error)}_pt_{pt_data['index']}_hospi_pt_{pt_data['pt_num']}_fed_{feeding_num}_fold_{fold_num}.png")
        # fig_path = os.path.join(folder_path, f"errors/all_patients/", f"error_{int(error)}_pt_{pt_data['index']}_hospi_pt_{pt_data['pt_num']}_fed_{feeding_num}_fold_{fold_num}.png")

        feeding_time_str = pt_data['feeding_start_time'].strftime("%Y-%m-%d %H:%M:%S")
        feeding_time_sanitized = feeding_time_str.replace(":", "-").replace(" ", "_")
        fig_path = os.path.join(folder_path, "errors", "all_patients",
                                f"error_{int(error)}_pt_idx_{pt_data['index']}_hospi_pt_{pt_data['hospital_name']}_{pt_data['pt_num']}_{feeding_time_sanitized}_feed_{feeding_num}.png")
    else:
        fig_path = os.path.join(folder_path, f"errors/all_patients/", f"error_{int(error)}_pt_{pt_data['index']}_fed_{feeding_num}.png")
    fig.savefig(fig_path)
    # fold_path = os.path.join(folder_path, f"errors/fold_errors/{PATIENT_TYPE}/fold_{fold_num}", f"{PATIENT_TYPE}_error_{int(error)}_pt_{pt_data['index']}_hospi_pt_{pt_data['pt_num']}_fed_{feeding_num}_fold_{fold_num}.png")
    # fig.savefig(fold_path)
    plt.close(fig) # Close the figure after collecting for saving


def simple_plot_single_patient_data(folder_path,  pt, pt_data, fold_num, error=None, prediction_info=None, feeding_num=1, D_0_TIME=1440, D_1_TIME=2880, PATIENT_TYPE=None, message=None):
    # print(f"{feeding_num=} | {error=}")
    # if error is None: return
    error = 0

    sns.set_style("whitegrid", {"grid.color": ".6", "grid.linestyle": ":"})
    # gray_color = '#f0f0f0'
    gray_color = '#ffffff'
    color = 'darkorange'

    actual_bst = pt_data['bst'][:D_1_TIME]
    mask = actual_bst != 0
    actual_bst = actual_bst[mask]
    if not pt_data.get(f"feeding_{feeding_num}"): return

    feeding_start, feeding_end = pt_data[f"feeding_{feeding_num}"]['initial_bst_idx'], pt_data[f"feeding_{feeding_num}"]["target_bst_idx"]
    # feeding_start, feeding_end = 1414, 1981

    fig = plt.figure(figsize=(12, 9))
    fontsize = 16
    ax = {}  # Dictionary to store axes for each subplot

    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])  # Define the ratio of heights for each row
    ax1 = plt.subplot(gs[0])

    ax1.plot(np.where(mask)[0], actual_bst, label='Actual BST', marker='x') #color='orange'
    ax1.plot(pt_data[f"feeding_{feeding_num}"]['initial_bst_idx'], pt_data[f"feeding_{feeding_num}"]['initial_bst'],
             label='Initial BST', linestyle='None',
             marker='p', markersize=15, color='black')
    ax1.plot(pt_data[f"feeding_{feeding_num}"]['target_bst_idx'], pt_data[f"feeding_{feeding_num}"]['target_bst'],
             label='Target BST', linestyle='None',
             marker='p', markersize=15, color='red')


    segment_info = None

    fig.suptitle(f'Hospital: {pt_data["hospital_name"]} | Patient No: {pt_data["pt_num"]} '
              f'\n  Feeding Start Time: {pt_data["feeding_start_time"]} '
              f'\n Index = {pt} | Feeding Num: {feeding_num} | Error: {error:.4f}'
              f'{segment_info if segment_info else ""}',
              # fontsize=8)
              fontsize=fontsize + 6)

    ax1.axvline(x=D_0_TIME, color='red', linestyle='--', label=f'Line at {D_0_TIME // 60} hours')


    step = 2 * 60  # 2 hours
    ax1.set_xticks(np.arange(0, D_1_TIME+1, step=step))
    xlabels = [str(i // 60) for i in range(0, D_1_TIME+1, step)]
    ax1.set_xticklabels(xlabels, rotation=45)
    # Explicitly set x-axis limits to remove extra spaces
    ax1.set_xlim(0, D_1_TIME)
    # ax1.set_xlabel('Time Steps (Hours)', fontsize=fontsize)
    ax1.set_ylabel('BGL', fontsize=fontsize)

    ax1.grid(True)

    # Create and plot subplots for other data (excluding ignored chart data)

    temporal_features = {
            'Calories (PN+EN)': pt_data['calorie'],
            # 'Total Calories': pt_data['calorie_a'],
            'Insulin': pt_data['insulin'],
            # 'Insulin': pt_data['insulin_a'],
            # 'Steroid': pt_data['steroid'],
            'PN': pt_data['tpn'],
            # 'Medication': pt_data['medication'],
            # 'Insulin-I': pt_data['insulin_i'],
            # 'Insulin-H': pt_data['insulin_h'],
            # 'Insulin-IV': pt_data['insulin_iv'],
            # 'Insulin-A': pt_data['insulin_a']
        }
    ignore_temporal_features = ['PN', 'Insulin-I', 'Insulin-H', 'Insulin-IV', 'Insulin-A']
    j = 2 if message is not None else 1
    for key in temporal_features.keys():
        if key in ignore_temporal_features: continue
        ax[key] = plt.subplot(gs[j], sharex=ax1)
        ax[key].plot(temporal_features[key][ :D_1_TIME], label=key if key!='Calories (PN+EN)' else 'PN+EN')
        if feeding_start !=0:
            if key!='Calories (PN+EN)':
                ax[key].plot( np.arange(feeding_start, feeding_end),temporal_features[key][feeding_start:feeding_end], color='darkorange', linewidth=3)
            else:
                # ax[key].plot(temporal_features['TPN / Fluid'][:D_1_TIME], label='Insulin-A', linestyle='dotted', color='yellow')
                ax[key].fill_between(range(D_1_TIME), temporal_features['PN'][:D_1_TIME], color=color, alpha=0.3, label='PN')
                ax[key].plot(np.arange(feeding_start, feeding_end), temporal_features[key][feeding_start:feeding_end], color='#BD457E', linewidth=3)#, label='PN+EN') # fuchsia, #E1AFD0, #BD457E
                ax[key].legend(loc='upper left', fontsize=fontsize)
        ax[key].axvline(x=D_0_TIME, color='red', linestyle='--')
        if key == 'Insulin':
            ax[key].set_xlabel('Time Steps (Hours)', fontsize=fontsize)

        ax[key].set_ylabel(key, fontsize=fontsize)
        ax[key].grid(True)
        j += 1
    plt.tight_layout(pad=2)

    if PATIENT_TYPE:
        feeding_time_str = pt_data['feeding_start_time'].strftime("%Y-%m-%d %H:%M:%S")
        feeding_time_sanitized = feeding_time_str.replace(":", "-").replace(" ", "_")
        # ORIGINAL
        # fig_path = os.path.join(folder_path, "errors", "all_patients",
        #                         f"error_{int(error)}_pt_idx_{pt_data['index']}_hospi_pt_{pt_data['hospital_name']}_{pt_data['pt_num']}_{feeding_time_sanitized}_feed_{feeding_num}.png")
        # MIMIC
        fig_path = os.path.join(folder_path, "errors", "all_patients",
                                f"error_{int(error)}_pt_idx_{pt_data['index']}_hospi_pt_{pt_data['hospital_name']}_feed_{feeding_num}.png")
    else:
        fig_path = os.path.join(folder_path, f"errors/all_patients/", f"error_{int(error)}_pt_{pt_data['index']}_fed_{feeding_num}.png")
    fig.savefig(fig_path)
    plt.close(fig)


# def plot_single_patient_data(folder_path,  pt, pt_data, fold_num, error=None, prediction_info=None, feeding_num=1, D_0_TIME=1440, D_1_TIME=2880, PATIENT_TYPE=None, message=None):
#     # print(f"{feeding_num=} | {error=}")
#     # if error is None: return
#     error = 0
#
#
#     sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
#     gray_color = '#f0f0f0'
#     color = 'darkorange'
#
#     if pt != pt_data["index"]:
#         print("++++++++++++++++++++++++++Problem++++++++++++++++++++++++++++")
#
#     actual_bst = pt_data['bst'][:D_1_TIME]
#     mask = actual_bst != 0
#     actual_bst = actual_bst[mask]
#     # for i, m in enumerate(mask):
#     #     if m == True:
#     #         print(i)
#     # feeding_start, feeding_end = 0, 0
#     feeding_start, feeding_end = 1414, 1981
#
#
#     fig = plt.figure(figsize=(12, 14))
#     fontsize = 16
#     ax = {}  # Dictionary to store axes for each subplot
#     if message is None:
#         gs = gridspec.GridSpec(5, 1, height_ratios=[2, 1, 1, 1, 1])  # Define the ratio of heights for each row
#         ax1 = plt.subplot(gs[0])
#     else:
#         gs = gridspec.GridSpec(6, 1, height_ratios=[1, 2, 1, 1, 1, 1])  # Define the ratio of heights for each row
#
#         # Display the Q-Value table
#         # Reset the index to make it a column, so we can display it in the table
#         message.reset_index(inplace=True)
#         message.columns = ['State (BGL, Calorie)'] + list(message.columns[1:])  # Rename columns to include the new "State" column
#         # message.columns = ['State'] + list(message.columns[1:])  # Rename columns to include the new "State" column
#         message.replace(to_replace=r'nan \(nan\)', value='', regex=True, inplace=True)  # Replace "nan (nan)" with an empty string
#         message.replace(to_replace=np.nan, value='', inplace=True)
#         ax0 = plt.subplot(gs[0])
#         ax0.axis('off')  # Hide the plot axes
#         table = ax0.table(cellText=message.values, colLabels=message.columns, cellLoc='center', loc='center')
#         table.auto_set_font_size(False)  # Disable automatic font scaling
#         table.set_fontsize(11)  # Set desired font size for the table
#         # Adjust the cell width and height
#         for (row, col), cell in table.get_celld().items():
#             if col == 0:
#                 cell.set_width(0.2)  # Set a wider width for the first column
#             else:
#                 cell.set_width(0.08)  # Set the standard width for other columns
#             cell.set_height(0.3)  # Set desired height for all cells
#         # ax0.set_title(" Q-Values Table", fontsize=12)
#
#         # Main plot: Plot actual BST and predictions (if available)
#         ax1 = plt.subplot(gs[1])
#     ax1.plot(np.where(mask)[0], actual_bst, label='Actual BST', marker='x') #color='orange'
#
#     # if feeding_start != 0:
#     #     indices = []
#     #     temp_bst = []
#     #     for m, b in zip(np.where(mask)[0], actual_bst):
#     #         if feeding_start<=m<=feeding_end:
#     #             indices.append(m)
#     #             temp_bst.append(b)
#     #     ax1.plot(indices, temp_bst, label='Actual BST', marker='x', color='orange')
#
#
#
#     segment_info = None
#     # Plot predictions if available
#     if prediction_info:
#         # ax1.plot(pt_data[f"feeding_{feeding_num}"]['target_bst_idx'], prediction_info, label='Predicted BST', linestyle='--', marker='^',
#         #          color='green')
#         ax1.plot(pt_data[f"feeding_{feeding_num}"]['initial_bst_idx'], pt_data[f"feeding_{feeding_num}"]['initial_bst'], label='Initial BST', linestyle='None',
#                  marker='o', color='black')
#         ax1.plot(pt_data[f"feeding_{feeding_num}"]['target_bst_idx'], pt_data[f"feeding_{feeding_num}"]['target_bst'], label='Target BST', linestyle='None',
#                  marker='D', color='red')
#
#         # seg_insulin = pt_data[f"feeding_{feeding_num}"]['seg_insulin']
#         # seg_calorie = pt_data[f"feeding_{feeding_num}"]['seg_calorie']
#         complete_insulin = pt_data[f"feeding_{feeding_num}"]['complete_insulin']
#         complete_calorie = pt_data[f"feeding_{feeding_num}"]['complete_calorie']
#         I_BST = pt_data[f'feeding_{feeding_num}']['initial_bst']
#         I_BST = pt_data[f'feeding_{feeding_num}']['initial_bst']
#         T_BST = pt_data[f"feeding_{feeding_num}"]['target_bst']
#         total_duration = pt_data[f"feeding_{feeding_num}"]['duration']
#         segment_info = (f"\n (Initial BST: {I_BST} | Calorie: {complete_calorie:.1f}) Target BGL: {T_BST}" +
#                         f"\n Hourly Insulin: Predicted= {prediction_info:.2f} |  Administered= {(complete_insulin*60.)/total_duration:.2f} ")
#         # segment_info=(f"\nInsulin # Complete:{complete_insulin:.2f} | Segment:{[f'{value:.2f}' for value in seg_insulin]}"+
#         #               f"\nCalorie # Complete:{complete_calorie:.2f} | Segment:{[f'{value:.2f}' for value in seg_calorie]}")
#
#     # Customize main plot
#
#     # print(message)
#
#     fig.suptitle(f'Hospital: {pt_data["hospital_name"]} | Patient No: {pt_data["pt_num"]} '
#               f'\n  Feeding Start Time: {pt_data["feeding_start_time"]} '
#               f'\n Index = {pt} | Feeding Num: {feeding_num} | Error: {error:.4f}'
#               f'{segment_info if segment_info else ""}',
#               # fontsize=8)
#               fontsize=fontsize + 6)
#
#     # ax1.set_title(f'Hospital: {pt_data["hospital_name"]} | Patient No: {pt_data["pt_num"]} '
#     #               f'\n  Feeding Start Time: {pt_data["feeding_start_time"]} '
#     #               f'\n Index = {pt} | Feeding Num: {feeding_num} | Error: {error:.4f}'
#     #               f'{segment_info if segment_info else ""}',
#     #               # fontsize=8)
#     #               fontsize=fontsize + 6)
#     # Custom subtitle (adjust coordinates as needed)
#
#     ax1.axvline(x=D_0_TIME, color='red', linestyle='--', label=f'Line at {D_0_TIME // 60} hours')
#     # for i, segment_line in enumerate(pt_data[f"feeding_{feeding_num}"]['segment_idx']):
#     #     if i == 0:
#     #         ax1.axvline(x=segment_line, color='gray', linestyle=(0, (1, 10)), label='Segments')  # 'loosely dotted',     (0, (1, 10)))
#     #     else:
#     #         ax1.axvline(x=segment_line, color='gray', linestyle=(0, (1, 10)))  # No label for subsequent lines
#
#     step = 2 * 60  # 2 hours
#     ax1.set_xticks(np.arange(0, D_1_TIME, step=step))
#     xlabels = [str(i // 60 + 1) for i in range(0, D_1_TIME, step)]
#     ax1.set_xticklabels(xlabels, rotation=45)
#     ax1.set_xlabel('Time Steps (Hours)', fontsize=fontsize)
#     ax1.set_ylabel('BST', fontsize=fontsize)
#     # legend = ax1.legend(loc='lower left', fontsize=fontsize)
#     # legend.get_frame().set_facecolor(gray_color)
#     ax1.grid(True)
#
#     # Create and plot subplots for other data (excluding ignored chart data)
#
#     temporal_features = {
#             'Total Calories': pt_data['calorie'],
#             # 'Total Calories': pt_data['calorie_a'],
#             'Insulin': pt_data['insulin'],
#             # 'Insulin': pt_data['insulin_a'],
#             'Steroid': pt_data['steroid'],
#
#             'TPN / Fluid': pt_data['tpn'],
#             'Medication': pt_data['medication'],
#             'Insulin-I': pt_data['insulin_i'],
#             'Insulin-H': pt_data['insulin_h'],
#             'Insulin-IV': pt_data['insulin_iv'],
#             'Insulin-A': pt_data['insulin_a']
#         }
#     ignore_temporal_features = ['TPN / Fluid', 'Insulin-I', 'Insulin-H', 'Insulin-IV', 'Insulin-A']
#     j = 2 if message is not None else 1
#     for key in temporal_features.keys():
#         # print(f"{j=}")
#         if key in ignore_temporal_features: continue
#         ax[key] = plt.subplot(gs[j], sharex=ax1)
#         # if key == 'Insulin':
#         # #     # ax[key].plot(all_features['Insulin-H'][pt, :D_1_TIME], label='Insulin-H', linestyle='--', color=color)
#         # #     # ax[key].plot(all_features['Insulin-I'][pt, :D_1_TIME], label='Insulin-I', linestyle='--', color='red')
#         # #     # ax[key].plot(temporal_features['Insulin-IV'][:D_1_TIME], label='Insulin-IV', linestyle='dotted', color=color)
#         #     ax[key].plot(temporal_features['Insulin-A'][:D_1_TIME], label='Insulin-A', linestyle='dotted', color=color)
#
#         ax[key].plot(temporal_features[key][ :D_1_TIME], label=key)
#         if feeding_start !=0:
#             if key!='Total Calories':
#                 ax[key].plot( np.arange(feeding_start, feeding_end),temporal_features[key][feeding_start:feeding_end], color='darkorange', linewidth=3)
#             else:
#                 ax[key].plot(np.arange(feeding_start, feeding_end), temporal_features[key][feeding_start:feeding_end], color='fuchsia', linewidth=3)
#         ax[key].axvline(x=D_0_TIME, color='red', linestyle='--')
#         ax[key].set_xlabel('Time Steps (Hours)', fontsize=fontsize)
#         ax[key].set_ylabel(key, fontsize=fontsize)
#         # legend = ax[key].legend(loc='lower left', fontsize=fontsize)
#         # legend.get_frame().set_facecolor(gray_color)
#         ax[key].grid(True)
#         j += 1
#
#     # Plot TPN/Fluid on twin axis of Total Calories
#     # key = 'TPN / Fluid'
#     # ax[key] = ax['Total Calories'].twinx()
#     # ax[key].plot(temporal_features[key][:D_1_TIME], color=color, label=key)
#     # legend = ax[key].legend(loc='lower right', fontsize=fontsize)
#     # legend.get_frame().set_facecolor(gray_color)
#     # plt.tight_layout(pad=2)
#
#     # Final adjustments and save figure
#     plt.tight_layout(pad=2)
#     if PATIENT_TYPE:
#         # fig_path = os.path.join(folder_path, f"errors/all_patients/", f"{PATIENT_TYPE}_error_{int(error)}_pt_{pt_data['index']}_hospi_pt_{pt_data['pt_num']}_fed_{feeding_num}_fold_{fold_num}.png")
#         # fig_path = os.path.join(folder_path, f"errors/all_patients/", f"error_{int(error)}_pt_{pt_data['index']}_hospi_pt_{pt_data['pt_num']}_fed_{feeding_num}_fold_{fold_num}.png")
#
#         feeding_time_str = pt_data['feeding_start_time'].strftime("%Y-%m-%d %H:%M:%S")
#         feeding_time_sanitized = feeding_time_str.replace(":", "-").replace(" ", "_")
#         fig_path = os.path.join(folder_path, "errors", "all_patients",
#                                 f"error_{int(error)}_pt_idx_{pt_data['index']}_hospi_pt_{pt_data['hospital_name']}_{pt_data['pt_num']}_{feeding_time_sanitized}_feed_{feeding_num}.png")
#     else:
#         fig_path = os.path.join(folder_path, f"errors/all_patients/", f"error_{int(error)}_pt_{pt_data['index']}_fed_{feeding_num}.png")
#     fig.savefig(fig_path)
#     # fold_path = os.path.join(folder_path, f"errors/fold_errors/{PATIENT_TYPE}/fold_{fold_num}", f"{PATIENT_TYPE}_error_{int(error)}_pt_{pt_data['index']}_hospi_pt_{pt_data['pt_num']}_fed_{feeding_num}_fold_{fold_num}.png")
#     # fig.savefig(fold_path)
#     plt.close(fig) # Close the figure after collecting for saving

