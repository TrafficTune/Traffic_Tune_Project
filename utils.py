import pandas as pd
import subprocess
import json
import os
import csv
from ray import tune


def send_imessage(message: str, recipient: str):
    """
    python script to send imessage when the training is done
    you can write in var message the message you want to send
    and in var recipient the recipient email or phone number with country code which connected to the icloud account
    for now, it's manual but we can use it as a function which will be called whenever we want
    """

    apple_script = f'''
    tell application "Messages"
        set targetService to 1st service whose service type = iMessage
        set targetBuddy to buddy "{recipient}" of targetService
        send "{message}" to targetBuddy
    end tell
    '''
    subprocess.run(['osascript', '-e', apple_script])


def save_custom_metrics_to_csv(results_list: list, num_intersection: int, experiment_type: str, cycle_index=-1):
    """
    Saves the custom metrics from RLlib results to a CSV file.

    Parameters:
    - results (list): The list of results from RLlib training.
    - num_intersection_to_train (int): The number of intersections being trained.
    - experiment_type (str): The type of experiment.

    Returns:
    - None
    """
    if results_list:
        result_grid = results_list[cycle_index]
        result = result_grid[0]
        custom_metrics = result.metrics

        # Print the custom metrics (optional)
        print(custom_metrics)

        # Create a DataFrame from the custom metrics
        df = pd.DataFrame(custom_metrics)

        # Construct the file path for saving the CSV
        file_path = f"Outputs/Training/intersection_{num_intersection}/experiments/{experiment_type}_intersection_{num_intersection}.csv"

        # Save the DataFrame to a CSV file
        df.to_csv(file_path, index=False)
        print(f"Custom metrics saved to {file_path}")
    else:
        print("No results available to save.")


def save_result_from_json(json_file_path: str, num_intersection: int, experiment_type: str, cycle_index=-1):
    """
    Saves the custom metrics from RLlib results to a CSV file.

    Parameters:
    - json_file_path (str): The path to the JSON file containing RLlib training results.
    - num_intersection (int): The number of intersections being trained.
    - experiment_type (str): The type of experiment.
    - cycle_index (int): The index of the cycle in the results, default is -1 for the last cycle.

    Returns:
    - None
    """
    with open(json_file_path, "r") as file:
        json_results = [json.loads(line) for line in file]

    if json_results:
        # Access the result of the specified cycle
        result_grid = json_results[cycle_index]

        # Create a DataFrame from the custom metrics
        df = pd.DataFrame([result_grid])

        # Construct the file path for saving the CSV
        file_path = f"Outputs/Training/intersection_{num_intersection}/experiments/{experiment_type}_intersection_{num_intersection}.csv"

        # Save the DataFrame to a CSV file
        df.to_csv(file_path, index=False)
        print(f"Custom metrics saved to {file_path}")
    else:
        print("No results available to save.")


def extract_and_write_all_params():
    """
    Extract parameters from JSON files for different reinforcement learning algorithms and write them to CSV files.

    This function walks through a directory structure, finds parameter files for PPO, DQN, and DDQN algorithms,
    extracts relevant parameters, and writes them to separate CSV files for each algorithm.

    The function performs the following steps:
    1. Defines the base path and parameters to extract for each algorithm.
    2. Walks through the directory structure to find 'params.json' files.
    3. Extracts intersection number and date-time information from the file path.
    4. Loads the full configuration from each 'params.json' file.
    5. Determines the algorithm (PPO, DQN, or DDQN) based on the file path.
    6. Extracts relevant parameters for the identified algorithm.
    7. Organizes the extracted data into rows for each algorithm.
    8. Sorts the rows by intersection number.
    9. Writes the extracted parameters to separate CSV files for each algorithm.

    The CSV files are saved in a 'params' subdirectory within the base path, with filenames like 'PPO_params.csv',
    'DQN_params.csv', and 'DDQN_params.csv'.

    Note:
    - The function assumes a specific directory structure and naming convention for the parameter files.
    - It handles potential errors during file processing and continues with the next file if an error occurs.
    - The intersection numbers are extracted from directory names starting with 'intersection_'.
    - The date-time information is extracted from directory names containing 'PPO_', 'DQN_', or 'DDQN_'.

    Raises:
        OSError: If there are issues creating the output directory or writing to files.
        json.JSONDecodeError: If there are issues parsing the JSON files.
    """
    base_path = 'Outputs/Training'

    # Define parameters to extract for each algorithm
    params_for_algorithms = {
        "PPO": [
            "lr",
            "gamma",
            "num_sgd_iter",
            "lambda_",
            "clip_param",
            "entropy_coeff"
        ],
        "DQN": [
            "lr",
            "gamma",
            "train_batch_size",
            "target_network_update_freq",
            "hiddens",
            "n_step",
            "adam_epsilon"
        ],
        "DDQN": [
            "lr",
            "gamma",
            "train_batch_size",
            "target_network_update_freq",
            "hiddens",
            "n_step",
            "adam_epsilon"
        ]
    }

    # Prepare a dictionary to store rows for each algorithm
    algorithm_rows = {"PPO": [], "DQN": [], "DDQN": []}

    # Iterate over all directories in base_path
    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            # Construct the path to params.json
            params_path = os.path.join(root, dir_name, 'params.json')
            if not os.path.exists(params_path):
                continue  # Skip if params.json does not exist

            try:
                # Extract the intersection number and date-time from the path
                path_parts = params_path.split('/')

                # Find intersection number
                intersection_part = next((part for part in path_parts if part.startswith('intersection_')), None)
                if intersection_part:
                    intersection_num = intersection_part.split('_')[1]
                else:
                    intersection_num = 'unknown'

                # Find date-time part
                date_time_part = next(
                    (part for part in path_parts if 'PPO_' in part or 'DQN_' in part or 'DDQN_' in part), None)
                if date_time_part:
                    date_time_parts = date_time_part.split('_')[1:3]  # Extract both parts [1] and [2]
                    date_time_str = '_'.join(date_time_parts)  # Combine them with an underscore
                else:
                    date_time_str = 'unknown'

                # Load the full configuration
                with open(params_path, 'r') as file:
                    full_config = json.load(file)

                # Determine the algorithm
                algorithm = 'PPO' if 'PPO_' in date_time_part else 'DQN' if 'DQN_' in date_time_part else 'DDQN' if 'DDQN_' in date_time_part else 'unknown'

                # Extract the relevant parameters for the algorithm
                if algorithm in params_for_algorithms:
                    params_to_extract = params_for_algorithms[algorithm]
                    extracted_params = {key: full_config[key] for key in params_to_extract if key in full_config}

                    # Prepare the row for CSV
                    row = {
                        "Intersection": f"intersection_{intersection_num}",
                        "DateTime": f"{algorithm}_{date_time_str}",
                        **extracted_params
                    }
                    algorithm_rows[algorithm].append(row)

            except Exception as e:
                print(f"Error processing path: {params_path}. Error: {e}")

    # Define the output directory for CSV files
    output_dir = os.path.join(base_path, 'params')
    os.makedirs(output_dir, exist_ok=True)

    # Write each algorithm's rows to a separate CSV file
    for algorithm, rows in algorithm_rows.items():
        if rows:
            # Sort rows by the intersection number
            rows.sort(
                key=lambda x: int(x["Intersection"].split('_')[1]) if x["Intersection"].split('_')[1].isdigit() else 0)

            # Define the output CSV file path for the current algorithm
            output_file_path = os.path.join(output_dir, f'{algorithm}_params.csv')

            # Write the rows to the CSV file
            with open(output_file_path, 'w', newline='') as file:
                fieldnames = ["Intersection", "DateTime"] + params_for_algorithms[algorithm]
                writer = csv.DictWriter(file, fieldnames=fieldnames)

                writer.writeheader()
                writer.writerows(rows)


def convert_to_tune_calls(param: dict):
    """
   Convert a dictionary of parameter specifications to Ray Tune search space calls.

   This function takes a dictionary where each key represents a parameter name and
   each value is a dictionary specifying a Ray Tune function and its arguments. It
   converts these specifications into the appropriate Ray Tune search space objects.

   Args:
       param (dict): A dictionary where keys are parameter names and values are dictionaries
                     containing:
                       - 'func' (str): The name of the Ray Tune function (e.g., 'tune.loguniform').
                       - 'args' (list): A list of arguments to be passed to the Ray Tune function.

   Returns:
       dict: A dictionary where keys are the same parameter names and values are the corresponding
             Ray Tune search space objects.
   """
    param_space = {}
    for key, value in param.items():
        if value['func'] == 'tune.loguniform':
            param_space[key] = tune.loguniform(*value['args'])
        elif value['func'] == 'tune.uniform':
            param_space[key] = tune.uniform(*value['args'])
        elif value['func'] == 'tune.choice':
            param_space[key] = tune.choice(*value['args'])
        elif value['func'] == 'tune.randint':
            param_space[key] = tune.randint(*value['args'])
    return param_space
