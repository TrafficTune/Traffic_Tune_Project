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
        print(f"Minimum waiting time: {min_waiting_time}")
    else:
        print("\nNo episodes were processed successfully.")

    return min_episode, overall_mean, overall_std

# You can add more functions here if needed
# main.py

# from episode_analysis import analyze_episodes
#
# # Base path for the CSV files
# base_path = "Outputs/Training/intersection_4/experiments/"
#
# # Call the function
# mean_waiting_times, overall_mean, overall_std = analyze_episodes(base_path, file_pattern="DQN_intersection_4_random_easy_1_07.29-13:05:23_conn0_ep{}.csv")
#
# # You can now use these returned values for further analysis or visualization if needed
# print(f"Returned mean waiting times: {mean_waiting_times}")
# print(f"Returned overall mean: {overall_mean}")
# print(f"Returned overall standard deviation: {overall_std}")


# plot_waiting_time.py


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

        # If you want to save the plot instead of showing it:
        # plt.savefig(f"episode_{episode_number}_waiting_time.png")
        # plt.close()
    else:
        print(f"File for episode {episode_number} not found: {full_path}")

    # # main_script.py
    #
    # from vm import plot_waiting_time
    #
    # # Base path for the CSV files
    # base_path = "/Users/md/Desktop/Traffic_Tune_Project/Outputs/Training/intersection_4/experiments/"
    #
    # # Choose an episode number
    # episode_number = 7  # You can change this to any episode number you want to plot
    #
    # # Call the function to plot the waiting time
    # plot_waiting_time(base_path, episode_number)


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
        # Access the result of the specified cycle
        result_grid = json_results[cycle_index]

        values = result_grid["env_runners"]["hist_stats"]["episode_reward"]
        # Print the custom metrics (optional)
        print(values)
        # Plotting the values
        plt.plot(values, marker='o')
        plt.title(title)
        plt.xlabel('Episode number')
        plt.ylabel('Reward value')
        plt.grid(True)
        plt.show()
