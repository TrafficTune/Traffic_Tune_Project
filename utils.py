import pandas as pd
import subprocess


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


def save_custom_metrics_to_csv(results_list, num_intersection_to_train, experiment_type, cycle_index=-1):
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
        file_path = f"Outputs/Training/intersection_{num_intersection_to_train}/experiments/{experiment_type}_intersection_{num_intersection_to_train}.csv"

        # Save the DataFrame to a CSV file
        df.to_csv(file_path, index=False)
        print(f"Custom metrics saved to {file_path}")
    else:
        print("No results available to save.")