import pandas as pd
import subprocess
import json


def send_imessage(message, recipient):
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


def save_custom_metrics_to_csv(results_list, num_intersection, experiment_type, cycle_index=-1):
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


def save_result_from_json(json_file_path, num_intersection, experiment_type, cycle_index=-1):
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



