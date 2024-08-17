# episode_analysis.py

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import json


def analyze_episodes(num_intersection, num_episodes=8,
                     path_to_episode="DQN_intersection_4_random_easy_1_07.29-13:05:23_conn0_ep{}.csv"):
    # List to store the mean waiting times for each episode
    episode_mean_waiting_times = []
    base_path = f"Outputs/Training/intersection_{num_intersection}/experiments/"
    # Run for specified number of episodes
    for episode in range(1, num_episodes + 1):
        # Construct the filename for each episode
        full_path = os.path.join(base_path, path_to_episode.format(episode))

        # Check if the file exists
        if os.path.exists(full_path):
            # Read the CSV file
            csv_df = pd.read_csv(full_path)

            # Calculate the mean waiting time for this episode
            episode_mean_waiting_time = np.mean(csv_df["system_mean_waiting_time"])

            # Append the result to the list
            episode_mean_waiting_times.append(episode_mean_waiting_time)

            print(f"Episode {episode} mean waiting time: {episode_mean_waiting_time}")
        else:
            print(f"File for episode {episode} not found: {full_path}")

    # Print the list of mean waiting times
    print("\nMean waiting times for all episodes:")
    print(episode_mean_waiting_times)

    # Calculate overall statistics
    if episode_mean_waiting_times:
        overall_mean = np.mean(episode_mean_waiting_times)
        overall_std = np.std(episode_mean_waiting_times)

        print(f"\nOverall mean across all episodes: {overall_mean}")
        print(f"Overall standard deviation across all episodes: {overall_std}")

        # Find the episode with the minimum waiting time
        min_waiting_time = min(episode_mean_waiting_times)
        min_episode = episode_mean_waiting_times.index(min_waiting_time) + 1  # +1 because episodes are 1-indexed
        print(f"\nEpisode with minimum waiting time: Episode {min_episode}")
        print(f"Minimum waiting time: {min_waiting_time}\n")
    else:
        print("\nNo episodes were processed successfully.")

    return min_episode, overall_mean, overall_std


def plot_waiting_time(num_intersection, episode_number, path_to_episode="DQN_intersection_4_random_easy_1_07.29-13:05:23_conn0_ep{}.csv"):
    """
    Plots the system mean waiting time for a specific episode.

    Parameters:
    - base_path (str): The base path to the directory containing the CSV files.
    - episode_number (int): The episode number to plot.

    Returns:
    - None: Displays or saves the plot.
    """
    # Construct the filename for the chosen episode
    base_path = f"Outputs/Training/intersection_{num_intersection}/experiments/"
    full_path = os.path.join(base_path, path_to_episode.format(episode_number))

    # Check if the file exists
    if os.path.exists(full_path):
        # Read the CSV file
        csv_df = pd.read_csv(full_path)

        # Calculate the overall mean waiting time
        overall_mean = csv_df["system_mean_waiting_time"].mean()

        # Create the plot
        plt.figure(figsize=(12, 6))

        # Plot the original data
        plt.plot(csv_df.index, csv_df["system_mean_waiting_time"], label='Waiting Time')

        # Plot the overall mean line
        plt.axhline(y=overall_mean, color='r', linestyle='--', label=f'Overall Mean ({overall_mean:.2f})')

        # Customize the plot
        plt.title(f"System Mean Waiting Time for Episode {episode_number}")
        plt.xlabel("Time Step")
        plt.ylabel("Mean Waiting Time")
        plt.grid(True)
        plt.legend()

        # Show the plot
        plt.tight_layout()
        plt.show()

    else:
        print(f"File for episode {episode_number} not found: {full_path}")


def plot_reward_from_json(json_file_path, title, cycle_index=-1):
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
        result_grid = json_results[cycle_index]
        values = result_grid["env_runners"]["hist_stats"]["episode_reward"]
        plt.figure(figsize=(12, 6))
        plt.plot(values, marker='o', linestyle='-', color='#31a354')
        plt.title("Reward Over Episodes", fontsize=24)
        plt.xlabel('Episode number', fontsize=14)
        plt.ylabel('Reward value', fontsize=14)
        plt.grid(True)
        plt.show()


import json
import matplotlib.pyplot as plt
import numpy as np

def plot_rewards_intersection_algo(json_file_paths, title, cycle_index=-1, num_intersection=1):
    """
    Plots the custom metrics from RLlib results from multiple JSON files.

    Parameters:
    - json_file_paths (list): A list of paths to the JSON files containing RLlib training results.
    - title (str): The title of the plot.
    - cycle_index (int): The index of the cycle in the results, default is -1 for the last cycle.

    Returns:
    - None
    """
    colors = ['#31a354', '#3182bd', '#e6550d', '#756bb1', '#636363']  # Green, Blue, Orange, Purple, Gray
    plt.figure(figsize=(12, 6))

    for i, json_file_path in enumerate(json_file_paths):
        with open(json_file_path, "r") as file:
            json_results = [json.loads(line) for line in file]
        label = json_file_path.split("/")[-2]

        if json_results:
            # Access the result of the specified cycle
            result_grid = json_results[cycle_index]

            # Extract values
            values = result_grid["env_runners"]["hist_stats"]["episode_reward"]
            print(f"Original values for {json_file_path}: {values}")

            # Ensure values is a NumPy array
            values = np.array(values, dtype=np.float64)
            print(f"Converted to NumPy array: {values}")

            # Calculate IQR to identify outliers
            Q1 = np.percentile(values, 35)
            Q3 = np.percentile(values, 90)
            IQR = Q3 - Q1
            print(f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")

            # Define outlier boundaries
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")

            # Identify outliers
            outliers = values[(values < lower_bound) | (values > upper_bound)]
            filtered_values = values[(values >= lower_bound) & (values <= upper_bound)]

            # Print the outliers
            if outliers.size > 0:
                print(f"Outliers for {json_file_path}:")
                print(outliers)
            else:
                print(f"No outliers found for {json_file_path}.")

            if "7" in json_file_path:
                # Plotting the filtered values with a different color for each file
                plt.plot(filtered_values, marker='o', color=colors[i % len(colors)], label=label)
            else:
                plt.plot(values, marker='o', linestyle='-', color=colors[i % len(colors)], label=label)

    plt.title(title)
    plt.xlabel('Episode number')
    plt.ylabel('Reward value')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{title}.png")
    plt.show()


def plot_rewards_from_multiple_jsons(json_file_paths, title, cycle_index=-1):
    """
    Plots the custom metrics from RLlib results from multiple JSON files in a 2x3 grid.

    Parameters:
    - json_file_paths (list): A list of 6 paths to the JSON files containing RLlib training results.
    - title (str): The title of the overall figure.
    - cycle_index (int): The index of the cycle in the results, default is -1 for the last cycle.

    Returns:
    - None
    """
    assert len(json_file_paths) == 6, "You must provide exactly 6 JSON file paths."

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()  # Flatten to make it easier to iterate over

    colors = ['#31a354', '#3182bd', '#e6550d', '#756bb1', '#636363', '#fdae6b']  # A list of colors for each plot

    for i, json_file_path in enumerate(json_file_paths):
        with open(json_file_path, "r") as file:
            json_results = [json.loads(line) for line in file]

        if json_results:
            result_grid = json_results[cycle_index]
            values = result_grid["env_runners"]["hist_stats"]["episode_reward"]

            ax = axes[i]
            ax.plot(values, marker='o', linestyle='-', color=colors[i])
            ax.set_title(f"Intersection {i + 1}", fontsize=16)
            ax.set_xlabel('Episode number', fontsize=10)
            ax.set_ylabel('Reward value', fontsize=10)
            ax.grid(True)

    plt.suptitle(title, fontsize=24)
    plt.subplots_adjust(wspace=0.3, hspace=0.4)  # Increase the space between plots
    plt.savefig(f"{title}.png")
    plt.show()


def plot_episode_mean_return(csv_file_path, title):
    """
    Plots the mean return for each episode from a CSV file.

    Parameters:
    - csv_file_path (str): The path to the CSV file containing the mean return values.
    - title (str): The title of the plot.

    Returns:
    - None: Displays the plot.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot the mean return values
    plt.plot(df["env_runners/episode_return_mean"], marker='o')

    # Customize the plot
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Mean Return")
    plt.grid(True)
    plt.show()


def create_param_table(json_files_2):
    rows = []

    for i, file_path in enumerate(json_files_2, start=1):
        with open(file_path, 'r') as f:
            data = json.load(f)

        if "DQN" in file_path:
            # Columns for DQN
            columns = ["Single Intersection", "lr", "gamma", "train batch size",
                       "target network update freq", "hiddens", "n step", "adam_epsilon"]

            # Extract DQN parameters
            lr = data.get("lr", None)
            gamma = data.get("gamma", None)
            train_batch_size = data.get("train_batch_size", None)
            target_network_update_freq = data.get("target_network_update_freq", None)
            hiddens = data.get("hiddens", None)
            n_step = data.get("n_step", None)
            adam_epsilon = data.get("adam_epsilon", None)

            # Add the row for DQN
            rows.append({
                "Single Intersection": i,
                "lr": lr,
                "gamma": gamma,
                "train batch size": train_batch_size,
                "target network update freq": target_network_update_freq,
                "hiddens": hiddens,
                "n step": n_step,
                "adam_epsilon": adam_epsilon
            })

        elif "PPO" in file_path:
            # Columns for PPO
            columns = ["Single Intersection", "lr", "gamma", "num_sgd_iter",
                       "lambda_", "clip_param", "vf_loss_coeff", "grad_clip", "entropy_coeff"]

            # Extract PPO parameters
            lr = data.get("lr", None)
            gamma = data.get("gamma", None)
            num_sgd_iter = data.get("num_sgd_iter", None)
            lambda_ = data.get("lambda_",None)
            clip_param = data.get("clip_param", None)
            vf_loss_coeff = data.get("vf_loss_coeff", None)
            grad_clip = data.get("grad_clip", None)
            entropy_coeff = data.get("entropy_coeff", None)

            # Add the row for PPO
            rows.append({
                "Single Intersection": i,
                "lr": lr,
                "gamma": gamma,
                "num_sgd_iter": num_sgd_iter,
                "lambda_": lambda_,
                "clip_param": clip_param,
                "vf_loss_coeff": vf_loss_coeff,
                "grad_clip": grad_clip,
                "entropy_coeff": entropy_coeff
            })

    # Convert the list of rows into a DataFrame
    df = pd.DataFrame(rows, columns=columns)

    return df


if __name__ == "__main__":
    json_list = []
    algo_name = "PPO"

    # for intersection in range(1, 7):
    #     json_list.append(f"/Users/eviat/Desktop/Final_Project/training_best_result/intersection_{intersection}/{algo_name}/params.json")
