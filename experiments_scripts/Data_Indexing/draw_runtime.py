import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ------------------------------------------------------------------------------
#  Basic Config
# ------------------------------------------------------------------------------
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_PATH, "..", "..")  # or "/your/path"

x_labels    = ['AlexNet', 'ResNet50', 'ResNet152', 'UNet', 'VGG16', 'Bert']
nn_models   = ["alexnet", "resnet50", "resnet152", "unet", "vgg16", "bert"]  # row indices 0..5
method_list = [0, 1, 2, 3, 4, 5, 6]  # columns 0..3

NUM_MODELS  = len(x_labels)   # 7
NUM_METHODS = len(method_list)

data    = np.zeros((NUM_MODELS, NUM_METHODS))

data2    = np.zeros((NUM_MODELS, NUM_METHODS))


model_to_row = { name: i for (i, name) in enumerate(nn_models) }

# Baseline logs use a different capitalization
baseline_nn_mapping = {
    "alexnet":   "Alexnet",
    "resnet50":  "ResNet50",
    "resnet152": "resnet152",
    "unet":      "unet",
    "vgg16":     "vgg16",
    "bert":      "bert"
}

# ------------------------------------------------------------------------------
# CSV Parsing
# ------------------------------------------------------------------------------
def read_optipim_sum(csv_path, flex_path):
    """Return {'our_input','our_compute','our_output'} from the 'sum' row (ignore extras)."""
    if not os.path.isfile(csv_path):
        return None
    df1 = pd.read_csv(csv_path)
    df2 = pd.read_csv(flex_path)
    combined = pd.merge(df1, df2, on='Layer Name', suffixes=('_1', '_2'))
    mask = (combined['32-optipim_1'] != 'Infeasible') & (combined['32-optipim_2'] != 'Infeasible')
    df_filtered = combined[mask].copy()
    df_filtered['32-runtime_1'] = pd.to_numeric(df_filtered['32-runtime_1'])

    total_runtime = df_filtered['32-runtime_1'].sum()
    return total_runtime

def fill_arrays_for_device(device_type, device_baseline_folder, arr):
    """
    For rows 0..5 => read from CSV for col0(OptiPIM), col1(Heur), col3(ASIC).
    """
    nb_optipim   = "32banks"

    for model in nn_models:  # (alexnet..bert)
        row_idx = model_to_row[model]

        flex_path = os.path.join(
            PROJECT_ROOT, "exp_results", "Data_Indexing", "final_results",
            "4", device_type, model, f"{nb_optipim}.csv"
        )
        
        # (A) flexible => col=0
        opti_csv = os.path.join(
            PROJECT_ROOT, "exp_results", "Data_Indexing", "final_results",
            "4", device_type, model, f"{nb_optipim}.csv"
        )
        opti_data = read_optipim_sum(opti_csv, flex_path)
        if opti_data:
            arr[row_idx, 0]    = opti_data

        opti_csv = os.path.join(
            PROJECT_ROOT, "exp_results", "Data_Indexing", "final_results",
            "0", device_type, model, f"{nb_optipim}.csv"
        )
        opti_data = read_optipim_sum(opti_csv, flex_path)
        if opti_data:
            arr[row_idx, 1]    = opti_data

        opti_csv = os.path.join(
            PROJECT_ROOT, "exp_results", "Data_Indexing", "final_results",
            "1", device_type, model, f"{nb_optipim}.csv"
        )
        opti_data = read_optipim_sum(opti_csv, flex_path)
        if opti_data:
            arr[row_idx, 2]    = opti_data

        opti_csv = os.path.join(
            PROJECT_ROOT, "exp_results", "Data_Indexing", "final_results",
            "2", device_type, model, f"{nb_optipim}.csv"
        )
        opti_data = read_optipim_sum(opti_csv, flex_path)
        if opti_data:
            arr[row_idx, 3]    = opti_data

        opti_csv = os.path.join(
            PROJECT_ROOT, "exp_results", "Data_Indexing", "final_results",
            "3", device_type, model, f"{nb_optipim}.csv"
        )
        opti_data = read_optipim_sum(opti_csv, flex_path)
        if opti_data:
            arr[row_idx, 4]    = opti_data

        opti_csv = os.path.join(
            PROJECT_ROOT, "exp_results", "Data_Indexing", "final_results",
            "5", device_type, model, f"{nb_optipim}.csv"
        )
        opti_data = read_optipim_sum(opti_csv, flex_path)
        if opti_data:
            arr[row_idx, 5]    = opti_data

        opti_csv = os.path.join(
            PROJECT_ROOT, "exp_results", "Data_Indexing", "final_results",
            "6", device_type, model, f"{nb_optipim}.csv"
        )
        opti_data = read_optipim_sum(opti_csv, flex_path)
        if opti_data:
            arr[row_idx, 6]    = opti_data


# ------------------------------------------------------------------------------
#  Fill columns from various data indexing functions
# ------------------------------------------------------------------------------
fill_arrays_for_device("PUM", "simdram_baseline", data)
fill_arrays_for_device("NBP", "hbm_baseline",    data2)


# ------------------------------------------------------------------------------
# Finish Data Extraction
# ------------------------------------------------------------------------------
print("[INFO] Data Extraction Finished for all models and methods")
print("[INFO] Drawing Data Indexing Figure.")

# ------------------------------------------------------------------------------
#                               Plotting   
# ------------------------------------------------------------------------------

# ----------------------------------------------------------------
# 2. Speed-up Calculations for SIMDRAM (subplot a)
# ----------------------------------------------------------------

# Calculate speed-up for each method in each neural network model
speedup_data = []
for data in data:
    max_value = max(data)
    speedup_data.append([val/max_value for val in data])

# ----------------------------------------------------------------
# 3. Speed-up Calculations for HBM-PIM (subplot c)
# ----------------------------------------------------------------

# Calculate speed-up for each method in each neural network model
speedup_data_2 = []
for data in data2:
    max_value = max(data)
    speedup_data_2.append([val/max_value for val in data])

# ----------------------------------------------------------------
# 4. Create the Figure
# ----------------------------------------------------------------
fig, (ax, ax3) = plt.subplots(
    2, 1, figsize=(12, 8),
    gridspec_kw={'hspace': 0.35}
)

# ----------------------------------------------------------------
# 5. Adjust the Spacing Between Groups and the Bar Width
#    (get extra spacing and smaller bars)
# ----------------------------------------------------------------
group_spacing = 1.1  # Controls how spread out each group is
width = 0.1         # Shrink the bar width

# Create new x positions that incorporate the group spacing
x = np.arange(len(x_labels)) * group_spacing

method_labels = ['Flexible', 'Comb0', 'Comb1', 'Comb2', 'Comb3', 'Comb4', 'Comb5']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

# ----------------------------------------------------------------
# Plot SIMDRAM speedup (ax)
# ----------------------------------------------------------------
for j in range(NUM_METHODS):
    offset = (j - NUM_METHODS / 2) * width + width / 2
    ax.bar(x + offset,
           [speedup_data[i][j] for i in range(len(x_labels))],
           width, label=method_labels[j], color=colors[j])

ax.set_xticks(x)
ax.set_xticklabels(x_labels, rotation=45, ha='right')
ax.set_ylabel('Speedup')
ax.set_title('SIMDRAM (PUM) Normalized Runtime by Indexing Function')
ax.legend(loc='upper right', fontsize=7)
ax.axhline(y=1.0, color='black', linestyle='--', linewidth=0.8)

# ----------------------------------------------------------------
# Plot HBM-PIM speedup (ax3)
# ----------------------------------------------------------------
for j in range(NUM_METHODS):
    offset = (j - NUM_METHODS / 2) * width + width / 2
    ax3.bar(x + offset,
            [speedup_data_2[i][j] for i in range(len(x_labels))],
            width, label=method_labels[j], color=colors[j])

ax3.set_xticks(x)
ax3.set_xticklabels(x_labels, rotation=45, ha='right')
ax3.set_ylabel('Normalized Runtime')
ax3.set_title('HBM-PIM (NBP) Normalized Runtime by Indexing Function')
ax3.legend(loc='upper right', fontsize=7)
ax3.axhline(y=1.0, color='black', linestyle='--', linewidth=0.8)


# Check the existence of the folder
if not os.path.exists(f"{PROJECT_ROOT}/exp_results/Data_Indexing"):
    os.makedirs(f"{PROJECT_ROOT}/exp_results/Data_Indexing")

# Save the figure
plt.savefig(f"{PROJECT_ROOT}/exp_results/Data_Indexing/Data_Indexing_Runtime.pdf", bbox_inches='tight', format="pdf")


