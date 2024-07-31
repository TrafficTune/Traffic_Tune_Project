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
        </style>
        """,
        unsafe_allow_html=True
    )

# Function to analyze episodes
def analyze_episodes(num_intersection, files):
    episode_mean_waiting_times = []
    for file in files:
        if file is not None:
            csv_df = pd.read_csv(file)
            episode_mean_waiting_time = np.mean(csv_df["system_mean_waiting_time"])
            episode_mean_waiting_times.append(episode_mean_waiting_time)
        else:
            st.warning("One or more files not provided.")

    if episode_mean_waiting_times:
        overall_mean = np.mean(episode_mean_waiting_times)
        overall_std = np.std(episode_mean_waiting_times)
        min_waiting_time = min(episode_mean_waiting_times)
        min_episode = episode_mean_waiting_times.index(min_waiting_time) + 1
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
        plt.title(f"System Mean Waiting Time", fontsize=16)
        plt.xlabel("Time Step", fontsize=14)
        plt.ylabel("Mean Waiting Time", fontsize=14)
        plt.grid(True)
        plt.legend()
        st.pyplot(plt)
    else:
        st.warning("File not provided.")

# Function to plot reward from JSON
def plot_reward_from_json(json_file, title, cycle_index=-1):
    if json_file is not None:
        stringio = StringIO(json_file.getvalue().decode("utf-8"))
        json_results = [json.loads(line) for line in stringio.readlines()]
        if json_results:
            result_grid = json_results[cycle_index]
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

# Main function
def main():
    set_custom_style()
    st.title("Episode Analysis and Visualization")

    st.sidebar.header("Analyze Episodes")
    num_intersection = st.sidebar.number_input("Intersection Number", min_value=1, max_value=10, value=4)
    files = st.sidebar.file_uploader("Upload CSV Files for Episodes", accept_multiple_files=True, type="csv", key="upload_files")

    if st.sidebar.button("Analyze Episodes"):
        if files:
            episode_mean_waiting_times, overall_mean, overall_std, min_episode, min_waiting_time = analyze_episodes(num_intersection, files)
            if episode_mean_waiting_times is not None:
                st.header("Analysis Results")
                st.subheader("Mean Waiting Times for All Episodes")
                with st.expander("View Mean Waiting Times"):
                    st.write(episode_mean_waiting_times)
                st.subheader("Overall Metrics")
                st.metric("Overall Mean Waiting Time", f"{overall_mean:.2f}")
                st.metric("Overall Standard Deviation", f"{overall_std:.2f}")
                st.metric(f"Minimum Waiting Time (Episode {min_episode})", f"{min_waiting_time:.2f}")
        else:
            st.warning("Please upload CSV files for analysis.")

    st.sidebar.header("Plot Waiting Time")
    file = st.sidebar.file_uploader("Upload CSV File for Waiting Time Plot", type="csv", key="upload_waiting_time")
    if st.sidebar.button("Plot Waiting Time"):
        if file:
            st.header("Waiting Time Plot")
            plot_waiting_time(file)
        else:
            st.warning("Please upload a CSV file to plot waiting time.")

    st.sidebar.header("Plot Reward from JSON")
    json_file = st.sidebar.file_uploader("Upload JSON File", type="json", key="upload_json")
    title = st.sidebar.text_input("Plot Title", "Episode Reward Plot")
    cycle_index = st.sidebar.number_input("Cycle Index", min_value=-1, max_value=100, value=-1)
    if st.sidebar.button("Plot Reward"):
        if json_file:
            st.header(f"Reward Plot: {title}")
            plot_reward_from_json(json_file, title, cycle_index)
        else:
            st.warning("Please upload a JSON file to plot rewards.")

if __name__ == "__main__":
    main()