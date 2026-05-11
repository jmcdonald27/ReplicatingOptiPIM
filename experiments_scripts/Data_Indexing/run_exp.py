from multiprocessing import Pool
import subprocess
from tqdm import tqdm
import re
import csv

import os
import subprocess
import time

from parser_trace import *

# Constant Definition
target_device_idx_name_map = {
    0 : "PUM",
    1 : "NBP"
}

storage_method_idx_name_map = {
    0 : "No_output",
    2 : "All"
}

estimation_method_idx_name_map = {
    0 : "our_method",
    1 : "COSA"
}

class OpPerf:
    def __init__(self, 
                 opType,
                 numMulCol = None,
                 numOutCol = None,
                 numFilCol = None,
                 numInCol = None,
                 numRedOutCol = None,
                 numRedCol = None,
                 numColPE = None,
                 numOutPE = None,
                 numInPE = None,
                 numPESys = None,
                 numOutSys = None,
                 numInSys = None):
        # A string store the loop type
        self.opType = opType
        
        # Number of multiplications in a column
        self.numMulCol = numMulCol
        
        # Number of Outputs in a column
        self.numOutCol = numOutCol
        
        # Number of Filters in a column
        self.numFilCol = numFilCol
        
        # Number of Inputs in a column
        self.numInCol = numInCol
        
        # Number of reductions for an output in a column
        self.numRedOutCol = numRedOutCol
        
        # Number of total reductions in a column
        self.numRedCol = numRedCol
        
        # Number of Columns in a PE
        self.numColPE = numColPE
        
        # Number of Outputs per PE
        self.numOutPE = numOutPE
        
        # Number of inputs per PE
        self.numInPE = numInPE
        
        # Number of PEs used in the system
        self.numPESys = numPESys
        
        # Number of output transmissions in the system
        self.numOutSys = numOutSys
        
        # Number of input loading needed in the system
        self.numInSys = numInSys
        
        # Output transmission cost
        self.outTransCost = 0
        
        # Input loading Cost
        self.inLoadingCost = 0
        
        # Filter loading cost for HBM-PIM
        self.numFilPELoading = 0
        
        # Final Overall Performance
        self.finalPerformance = 0
    
    def print_variables(self):
        for key, value in vars(self).items():
            print(f"{key}: {value}")

## Result folder structure

# Environment Setup
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_PATH + "/../.."
NN_MODELS_PATH = f"{PROJECT_ROOT}/nn_models"
MLIR_BIN_PATH = f"{PROJECT_ROOT}/build/bin"


nn_models_list = ["alexnet", "resnet50", "resnet152", "unet", "vgg16", "bert"]
num_banks_list = ["32banks"]
target_device_list = [0, 1]
storage_method_list = [0]
estimation_method_list = [0, 1]
indexing_functions = [0, 1, 2, 3, 4, 5, 6]

def run_gurobi(args):
    gurobi_cmd, gurobi_workdir = args
    with open(f"{gurobi_workdir}/run_gurobi.log", "w") as output_f:
        start_time = time.time()
        try:
            subprocess.run(gurobi_cmd, shell=True, check=True, timeout=500, cwd=gurobi_workdir, stdout=output_f)
            end_time = time.time()
            runtime = end_time - start_time

            # Save runtime to a separate file
            with open(f"{gurobi_workdir}/runtime.log", "w") as runtime_f:
                runtime_f.write(f"Runtime: {runtime:.4f} seconds\n")

            return f"Success: {gurobi_workdir}, Runtime: {runtime:.4f}s"
        except subprocess.TimeoutExpired:
            end_time = time.time()
            runtime = end_time - start_time
            with open(f"{gurobi_workdir}/runtime.log", "w") as runtime_f:
                runtime_f.write(f"Runtime: {runtime:.4f} seconds (Timeout)\n")
            return f"Timeout: {gurobi_workdir}"
        except subprocess.CalledProcessError as e:
            end_time = time.time()
            runtime = end_time - start_time
            with open(f"{gurobi_workdir}/runtime.log", "w") as runtime_f:
                runtime_f.write(f"Runtime: {runtime:.4f} seconds (Failed)\n")
            return f"Failed: {gurobi_workdir}, Error: {str(e)}"

def parse_runtime(file_path):
    try:
        with open(file_path, 'r') as f:
            line = f.readline().strip()
            # Extract the number from "Runtime: X.XXXX seconds"
            return float(line.split()[1])
    except (FileNotFoundError, IndexError, ValueError):
        return None

def main_optipim():
    print("#######################################################################")
    print("#####################   OPTIPIM EXPERIMENT Fig10 ######################")
    print("#######################################################################")
    print()
    print("[WARNING] !! CHECK PARAMETERS !!")
    
    gurobi_cmds = []
    gurobi_folder = []
    
    for device_type in target_device_list:
        for nn_model in nn_models_list:
            for num_banks in num_banks_list:
                for sel_storage in storage_method_list:
                    for sel_estimation in estimation_method_list:
                        for indexing_function in indexing_functions:
                            # nn_model source path
                            model_path = f"{PROJECT_ROOT}/nn_models/{nn_model}/single_layers/{num_banks}"
                            
                            # Get all the layers in the folder
                            layers_to_run = [file for file in os.listdir(model_path)]
                            
                            for sel_layer in layers_to_run:
                                # Create the corresponding folder
                                folder_path = f"{PROJECT_ROOT}/exp_results/Data_Indexing/{indexing_function}/{target_device_idx_name_map[device_type]}/{nn_model}/{num_banks}/{storage_method_idx_name_map[sel_storage]}/{estimation_method_idx_name_map[sel_estimation]}"
                                folder_path = os.path.join(folder_path, sel_layer)
                                

                                if not os.path.exists(folder_path):
                                    os.makedirs(folder_path)
                                    
                                # Get test path
                                nn_layer = os.path.join(model_path, sel_layer)
                                
                                # Check the existence of the result file
                                if not os.path.exists(os.path.join(folder_path, "gurobi_output.txt")):
                                    gurobi_cmd = f"{MLIR_BIN_PATH}/pim-opt --data-layout-pass \
                                                    --trace-output-path {folder_path}/gurobi_output.txt \
                                                    --model-debug-file {folder_path}/milp.lp \
                                                    --trans-coeff-method {indexing_function} \
                                                    --target-device-type {device_type} \
                                                    --number-counting-method {sel_estimation} \
                                                    --obj-method 0 \
                                                    --storage-method {sel_storage} \
                                                    --config-arch-path {PROJECT_ROOT}/data/hbm_pim.json \
                                                    --config-knobs-path {PROJECT_ROOT}/data/knobs.json \
                                                    {nn_layer}"
                                                    
                                    # Store the command
                                    with open(os.path.join(folder_path, "gurobi_cmd.log"), "w") as f:
                                        f.write(gurobi_cmd)
                                    
                                    gurobi_cmds.append(gurobi_cmd)
                                    gurobi_folder.append(folder_path)
    
    assert len(gurobi_cmds) == len(gurobi_folder)
    gurobi_tasks = [(cmd, workdir) for cmd, workdir in zip(gurobi_cmds, gurobi_folder)]
    gurobi_tasks = gurobi_tasks

    print(f"[INFO] Total gurobi tasks: {len(gurobi_tasks)}")
    print(f"[INFO] Run Gurobi")
    
    print(f"[INFO] example gurobi work dir", gurobi_folder[0])
    
    with Pool(processes=os.cpu_count() // 4) as pool:
        for _ in tqdm(pool.imap_unordered(run_gurobi, gurobi_tasks), total=len(gurobi_tasks)):
            pass    

        
def get_first_line(file_path):
    try:
        return open(file_path).readline().strip().lower()
    except(FileNotFoundError, StopIteration):
        print(f"File {file_path} not found or is empty")
        return "Error"
        

def custom_sort_key(key):
    match = re.match(r"([a-zA-Z]+)(\d+)", key)
    if match:
        prefix = match.group(1)
        number = int(match.group(2))
        return (prefix, number)
    else:
        return (key, 0) 

def parse_result_optipim():
    print("############################################################################")
    print("##################### OPTIPIM RESULT EXTRACTION Fig 10 #####################")
    print("############################################################################")
    print()
    print("[WARNING] !! CHECK PARAMETERS !!")
    
    # Global information
    infeasible_configs = []
    
    # Iterate over all related sweeping aspects
    for device_type in target_device_list:
        for nn_model in nn_models_list:
            for indexing_function in indexing_functions:
            
                # Construct the path to store the output csv file
                folder_path = f"{PROJECT_ROOT}/exp_results/Data_Indexing/final_results/{indexing_function}/{target_device_idx_name_map[device_type]}/{nn_model}"
                
                # Check and create the corresponding folder
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                    
                # Iterate over all layers in a model 
                # Store the results of different configurations
                # Format of the final csv file for a single layer
                # Layer Name, 32-our, 32-cosa-est
                ## Iterate over all related configurations
                for num_banks in num_banks_list:
                    # Create the storing structure
                    layer_results = {}
                    
                    for sel_storage in storage_method_list:
                        for sel_estimation in estimation_method_list:
                            # Get sweep results folder path
                            sweep_results_folder = f"{PROJECT_ROOT}/exp_results/Data_Indexing/{indexing_function}/{target_device_idx_name_map[device_type]}/{nn_model}/{num_banks}/{storage_method_idx_name_map[sel_storage]}/{estimation_method_idx_name_map[sel_estimation]}"
                            layers_to_parse = [file for file in os.listdir(sweep_results_folder)]
                            
                            # Iterate over all layers
                            for sel_layer in layers_to_parse:
                                # Get the actual result  file path
                                gurobi_log_path = os.path.join(sweep_results_folder, sel_layer)
                                gurobi_log_path = os.path.join(gurobi_log_path, "run_gurobi.log")
                                
                                # Get the actual layer name
                                layer_name = sel_layer.split(".")[-3]
                                
                                # Parse the log file
                                found_optimal = False
                                time_out_flag = False
                                with open(gurobi_log_path, 'r') as file:
                                    for line in file:
                                        # If the MILP found the optimal solution
                                        if 'Optimal solution found' in line:
                                            found_optimal = True
                                            continue
                                    
                                        # If the MILP reached time out
                                        if 'Time limit reached' in line:
                                            time_out_flag = True
                                            continue
                                        
                                        # If the model is not feasible
                                        if 'Model is infeasible' in line:
                                            formated_value = "Infeasible"
                                            infeasible_configs.append(gurobi_log_path)
                                            
                                            if layer_name in layer_results:
                                                layer_results[layer_name].append(formated_value)
                                            elif layer_name not in layer_results:
                                                layer_results[layer_name] = ["Infeasible"]
                                            
                                            break
                                    
                                        if (found_optimal or time_out_flag) and 'Best objective' in line:
                                            # Extract the value after "Best Objective"
                                            line = line.strip()
                                            parts = line.split(",")
                                            first_part = parts[0]
                                            tokens = first_part.split()
                                            if len(tokens) >= 3:
                                                value = tokens[2]
                                                # Convert the value to a float and then format it to remove unnecessary zeros
                                                number = float(value)
                                                formated_value = '{:.6e}'.format(number)
                                                
                                                runtime_log_path = os.path.join(sweep_results_folder, sel_layer, "runtime.log")
                                                runtime = parse_runtime(runtime_log_path)

                                                if layer_name in layer_results:
                                                    layer_results[layer_name].append(formated_value)
                                                    layer_results[layer_name].append(runtime)
                                                elif layer_name not in layer_results:
                                                    layer_results[layer_name] = [formated_value, runtime]
                                            
                                            break
                            
                    # Generate the final csv file
                    csv_file = os.path.join(folder_path, num_banks + ".csv")
                    num_bank_value = num_banks.strip("banks")
                    header = ["Layer Name", f"{num_bank_value}-optipim", f"{num_bank_value}-runtime", 
                            f"{num_bank_value}-cosa_estimation_method", f"{num_bank_value}-cosa-runtime"]                    
                    with open(csv_file, mode='w', newline='') as csv_file:
                        writer = csv.writer(csv_file)
                        
                        # Write the header
                        writer.writerow(header)
                        
                        # Write the data
                        for layer_name, values in sorted(layer_results.items(), key=lambda item: custom_sort_key(item[0])):
                            writer.writerow([layer_name] + values)
                        
    print("[INFO] Done")
    
def parse_gurobi_log_file_optipim(file_path):
    # Create the storage structure
    layer_perf = OpPerf("conv")
    
    found_optimal = False
    
    
    # Parse the file
    with open(file_path, 'r') as file:
        for line in file:
            # If the model is infeasible
            if 'Model is infeasible' in line:
                formated_value = "Infeasible"
                layer_perf.finalPerformance = formated_value
                return layer_perf
            
            # If the MILP found the optimal solution
            if 'Optimal solution found' in line:
                found_optimal = True
                continue
            
            # If time out reached
            if 'Time limit reached' in line:
                formated_value = "Timeout"
                layer_perf.finalPerformance = formated_value
                return layer_perf
            
            # If needed value found
            if 'numOutSys' in line:
                line = line.strip()
                parts = line.split(":")
                value = parts[-1].strip()
                
                number = float(value)
                formated_number = '{:.6e}'.format(number)
                
                layer_perf.outTransCost = formated_number
                
            if 'numInSys' in line:
                line = line.strip()
                parts = line.split(":")
                value = parts[-1].strip()
                
                number = float(value)
                formated_number = '{:.6e}'.format(number)
                
                layer_perf.inLoadingCost = formated_number
                
            if 'numFilLoading' in line:
                line = line.strip()
                parts = line.split(":")
                value = parts[-1].strip()
                
                number = float(value)
                formated_number = '{:.6e}'.format(number)
                
                layer_perf.numFilPELoading = formated_number
                
            if (found_optimal) and 'Best objective' in line:
                line = line.strip()
                parts = line.split(",")
                first_part = parts[0]
                tokens = first_part.split()
                if len(tokens) >= 3:
                    value = tokens[2]
                    # Convert the value to a float and then format it to remove unnecessary zeros
                    number = float(value)
                    formated_number = '{:.6e}'.format(number)
                
                    layer_perf.finalPerformance = formated_number
    
    return layer_perf
    
def parse_result_optipim_breakdown():
    print()
    print("[INFO] Extract Detailed Breakdown for OPTIPIM!")
    
    
    # Iterate over all related sweeping aspects
    for device_type in target_device_list:
        for nn_model in nn_models_list:
            for indexing_function in indexing_functions:

            
                # Construct the path to store the output csv file
                folder_path = f"{PROJECT_ROOT}/exp_results/Data_Indexing/final_results/{indexing_function}/{target_device_idx_name_map[device_type]}/{nn_model}"
                
                # Check and create the corresponding folder
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                    
                # Iterate over all layers in a model 
                # Store the results of different configurations
                # Format of the final csv file for a single layer
                # Layer Name, 32-our-input, 32-our-computation, 32-our-reduction, 32-cosa-input, 32-cosa-computation, 32-cosa-reduction
                ## Iterate over all related configurations
                for num_banks in num_banks_list:
                    # Create the storing structure
                    layer_results = {}
                    header = ["Layer"]
                    layer_results["sum"] = [0, 0, 0, 0, 0, 0]
                    counter = 0
                    
                    for sel_storage in storage_method_list:
                        inner_counter = 0
                        for sel_estimation in estimation_method_list:
                            # Get sweep results folder path
                            sweep_results_folder = f"{PROJECT_ROOT}/exp_results/Data_Indexing/{indexing_function}/{target_device_idx_name_map[device_type]}/{nn_model}/{num_banks}/{storage_method_idx_name_map[sel_storage]}/{estimation_method_idx_name_map[sel_estimation]}"
                            layers_to_parse = [file for file in os.listdir(sweep_results_folder)]
                            layer_count = 0
                            
                            # Iterate over all layers
                            for sel_layer in layers_to_parse:
                                # Get the actual result  file path
                                gurobi_log_path = os.path.join(sweep_results_folder, sel_layer)
                                gurobi_log_path = os.path.join(gurobi_log_path, "run_gurobi.log")
                                
                                # Get the actual layer name
                                layer_name = sel_layer.split(".")[-3]
                                
                                # Calculate the layer breakdown
                                tmp_layer_perf = parse_gurobi_log_file_optipim(gurobi_log_path)
                                
                                # Get the desired cost
                                if (tmp_layer_perf.finalPerformance == "Infeasible"):
                                    tmp_in_loading = "Infeasible"
                                    tmp_out_reduction = "Infeasible"
                                    tmp_comp = "Infeasible"
                                elif (tmp_layer_perf.finalPerformance == "Timeout"):
                                    tmp_in_loading = "Timeout"
                                    tmp_out_reduction = "Timeout"
                                    tmp_comp = "Timeout"
                                else:
                                    tmp_in_loading = tmp_layer_perf.inLoadingCost
                                    tmp_out_reduction = tmp_layer_perf.outTransCost
                                    tmp_comp = float(tmp_layer_perf.finalPerformance) - float(tmp_in_loading) - float(tmp_out_reduction)
                                
                                # Parse the gurobi log file
                                if layer_name in layer_results:
                                    layer_results[layer_name].append(tmp_in_loading)
                                    layer_results[layer_name].append(tmp_comp)
                                    layer_results[layer_name].append(tmp_out_reduction)
                                elif layer_name not in layer_results:
                                    layer_results[layer_name] = [tmp_in_loading, tmp_comp, tmp_out_reduction]
                                
                                # Update counter
                                if (tmp_in_loading != "Infeasible" and tmp_comp != "Infeasible" and tmp_out_reduction != "Infeasible"):
                                    if (tmp_in_loading != "Timeout" and tmp_comp != "Timeout" and tmp_out_reduction != "Timeout"):
                                        
                                        layer_results["sum"][counter * 6 + inner_counter * 3 ] += float(tmp_in_loading)
                                        layer_results["sum"][counter * 6 + inner_counter * 3 + 1] += float(tmp_comp)
                                        layer_results["sum"][counter * 6 + inner_counter * 3 + 2] += float(tmp_out_reduction)
                                
                                layer_count += 1
                            
                            # Update header
                            num_bank_value = num_banks.strip("banks")
                            header.append(f"{num_bank_value}-{estimation_method_idx_name_map[sel_estimation]}-input_loading")
                            header.append(f"{num_bank_value}-{estimation_method_idx_name_map[sel_estimation]}-compute")
                            header.append(f"{num_bank_value}-{estimation_method_idx_name_map[sel_estimation]}-out_reduction")
                            
                            # Update counter
                            inner_counter += 1
                            
                        # Update counter
                        counter += 1
                    
                    # Generate the final csv file
                    csv_file = os.path.join(folder_path, num_banks + "_breakdown.csv")
                    
                    with open(csv_file, mode='w', newline='') as csv_file:
                        writer = csv.writer(csv_file)
                        
                        # Add header
                        writer.writerow(header)
                        
                        # Write the data
                        for layer_name, values in sorted(layer_results.items(), key=lambda item: custom_sort_key(item[0])):
                            writer.writerow([layer_name] + values)
                        
    print("[INFO] Done")


if __name__ == "__main__":
    # [Step 1]: Run experiments 
    main_optipim()
    
    # [Step 2]: Parse all the results
    parse_result_optipim()
    parse_result_optipim_breakdown()

