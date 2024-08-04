import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from io import StringIO


# Define a function for custom styling
def set_custom_style():
    st.markdown(
        """
        <style>
        .stMetric {
            font-size: 18px;
            font-weight: bold;
        }
        .stPlotlyChart, .stPyplot {
            margin-top: 20px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 24px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 12px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .analysis-box {
            background-color: #f9f9f9;
            padding: 10px;
            border-radius: 10px;
            border: 1px solid #ddd;
            margin-top: 10px;
        }
        .image-container {
            position: fixed;
            bottom: 0;
            right: 0;
            width: 100%;
            text-align: right;
            padding: 30px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="image-container">
            <img src="https://github.com/TrafficTune/Traffic_Tune_Project/assets/73496652/c71694bb-5653-4a4e-b20a-05a5c9d263a5" alt="Placeholder Image">
        </div>
        """,
        unsafe_allow_html=True
    )


def extract_key_from_filename(filename):
    # Split by underscore and extract the required parts for the key
    parts = filename.split("_")
    if len(parts) == 8:
        level = f"{parts[3]}_{parts[4]}"
    elif len(parts) == 9:
        level = f"{parts[3]}_{parts[4]}_{parts[5]}"
    episode = parts[-1].split("ep")[1].split(".")[0]
    return f"{level}_{episode}"


# Function to analyze episodes
def analyze_episodes(files):
    episode_mean_waiting_times = {}

    for file in files:
        if file is not None:
            csv_df = pd.read_csv(file)
            episode_mean_waiting_time = np.mean(csv_df["system_mean_waiting_time"])
            key = extract_key_from_filename(file.name)
            episode_mean_waiting_times[key] = episode_mean_waiting_time
        else:
            st.warning("One or more files not provided.")

    if episode_mean_waiting_times:
        overall_mean = np.mean(list(episode_mean_waiting_times.values()))
        overall_std = np.std(list(episode_mean_waiting_times.values()))
        min_waiting_time = min(episode_mean_waiting_times.values())
        min_episode = min(episode_mean_waiting_times, key=episode_mean_waiting_times.get)
        return episode_mean_waiting_times, overall_mean, overall_std, min_episode, min_waiting_time
    else:
        st.warning("No episodes were processed successfully.")
        return None, None, None, None, None


# Function to plot waiting time
def plot_waiting_time(file):
    if file is not None:
        csv_df = pd.read_csv(file)
        overall_mean = csv_df["system_mean_waiting_time"].mean()
        plt.figure(figsize=(12, 6))
        plt.plot(csv_df.index, csv_df["system_mean_waiting_time"], label='Waiting Time', color='#2b8cbe')
        plt.axhline(y=overall_mean, color='r', linestyle='--', label=f'Overall Mean ({overall_mean:.2f})')
        plt.title("System Mean Waiting Time", fontsize=16)
        plt.xlabel("Time Step", fontsize=14)
        plt.ylabel("Mean Waiting Time", fontsize=14)
        plt.grid(True)
        plt.legend()
        st.pyplot(plt)
    else:
        st.warning("File not provided.")


# Function to plot reward from JSON
def plot_reward_from_json(json_file, title, episode_num=0):
    if json_file is not None:
        stringio = StringIO(json_file.getvalue().decode("utf-8"))
        json_results = [json.loads(line) for line in stringio.readlines()]
        if json_results:
            result_grid = json_results[episode_num]
            values = result_grid["env_runners"]["hist_stats"]["episode_reward"]
            plt.figure(figsize=(12, 6))
            plt.plot(values, marker='o', linestyle='-', color='#31a354')
            plt.title(title, fontsize=16)
            plt.xlabel('Episode number', fontsize=14)
            plt.ylabel('Reward value', fontsize=14)
            plt.grid(True)
            st.pyplot(plt)
        else:
            st.warning("No data available in the JSON file.")
    else:
        st.warning("JSON file not provided.")


def plot_episode_mean_return(csv_file, title):
    if csv_file is not None:
        # Read CSV content
        stringio = StringIO(csv_file.getvalue().decode("utf-8"))
        df = pd.read_csv(stringio)
        df = df.dropna(subset=['env_runners/episode_return_mean'])

        # Assuming the column you want to plot is named 'env_runners/episode_return_mean'
        if 'env_runners/episode_return_mean' in df.columns:
            values = df['env_runners/episode_return_mean']

            # Plot the values
            plt.figure(figsize=(12, 6))
            plt.plot(values, marker='o', linestyle='-', color='#31a354')
            plt.title(title, fontsize=16)
            plt.xlabel('Episode number', fontsize=14)
            plt.ylabel("Mean Return", fontsize=14)
            plt.grid(True)
            st.pyplot(plt)
        else:
            st.warning("The column 'env_runners/episode_return_mean' is not available in the CSV file.")
    else:
        st.warning("CSV file not provided.")


# Main function
def main():
    set_custom_style()
    st.markdown("### ***Analysis and Visualizations for Traffic Tune Experiments***")
    st.markdown("> Upload your desired files")

    # File uploaders and input fields in the sidebar
    plot_title = st.sidebar.text_input("Plot Title", "Episode Return Plot")
    st.sidebar.header("Upload Files for Analysis")
    csv_files = st.sidebar.file_uploader("Upload Episodes CSV Files", accept_multiple_files=True, type="csv",
                                         key="upload_files")
    progress_csv_file = st.sidebar.file_uploader("Upload Progress CSV File", type="csv", key="upload_csv_progress")
    json_file = st.sidebar.file_uploader("Upload Return JSON File", type="json", key="upload_json")
    episode_num = st.sidebar.number_input("Episode Index", min_value=-1, max_value=100, value=-1)
    csv_file = st.sidebar.file_uploader("Upload Episode CSV File", type="csv", key="upload_episode_csv")

    # Analysis and plots
    st.markdown("#### Analysis and Plots")

    # Analyze episodes
    if csv_files:
        episode_mean_waiting_times, overall_mean, overall_std, min_episode, min_waiting_time = (
            analyze_episodes(csv_files))
        if episode_mean_waiting_times is not None:
            st.subheader("Mean Waiting Times for All Episodes")
            with st.expander("View Mean Waiting Times"):
                episode_mean_waiting_times_with_numbers = {f"{k}": v for k, v in episode_mean_waiting_times.items()}
                st.write(episode_mean_waiting_times_with_numbers)

            st.subheader("Overall Metrics")
            st.metric("Overall Mean Waiting Time", f"{overall_mean:.2f}")
            st.metric("Overall Standard Deviation", f"{overall_std:.2f}")
            st.metric(f"Minimum Waiting Time (Episode {min_episode})", f"{min_waiting_time:.2f}")
        else:
            st.warning("Please upload CSV files for analysis.")
    else:
        st.warning("* Please upload Episodes CSV files for analysis")

    # Plot return from CSV
    if progress_csv_file:
        st.subheader(f"Return Plot: {plot_title}")
        plot_episode_mean_return(progress_csv_file, plot_title)
    else:
        st.warning("* Please upload a CSV progress file to plot returns")

    # Plot reward from JSON
    if json_file:
        st.subheader(f"Reward Plot: {plot_title}")
        plot_reward_from_json(json_file, plot_title, episode_num)
    else:
        st.warning("* Please upload a JSON file to plot rewards")

    # Plot reward from JSON
    if csv_file:
        st.subheader(f"Episode Waiting Time Plot: {plot_title}")
        plot_waiting_time(csv_file)
    else:
        st.warning("* Please upload a JSON file to plot rewards")


if __name__ == "__main__":
    main()
