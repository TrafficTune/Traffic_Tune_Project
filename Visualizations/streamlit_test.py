import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from io import StringIO
from matplotlib.backends.backend_pdf import PdfPages


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


# Function to extract key from filename
def extract_key_from_filename(filename):
    parts = filename.split("_")
    level = "Default"
    if len(parts) == 7:
        level = f"{parts[2]}_{parts[3]}"
    elif len(parts) == 8:
        if parts[2] != "random":
            level = f"Hour {parts[2]}-{parts[3]}"
        else:
            level = f"{parts[2]}_{parts[3]}_{parts[4]}"
    elif len(parts) == 9:
        level = f"{parts[2]}_{parts[3]}_{parts[4]}_{parts[5]}"

    episode = parts[-1].split("ep")[1].split(".")[0]

    return f"{episode} in {level} Route file "


# Function to analyze episodes
def analyze_episodes(files):
    episode_mean_waiting_times = {}
    episode_files = {}

    for file in files:
        if file is not None:
            csv_df = pd.read_csv(file)
            episode_mean_waiting_time = np.mean(csv_df["system_mean_waiting_time"])
            key = extract_key_from_filename(file.name)
            episode_mean_waiting_times[key] = episode_mean_waiting_time
            episode_files[key] = csv_df
        else:
            st.warning("One or more files not provided.")

    if episode_mean_waiting_times:
        overall_mean = np.mean(list(episode_mean_waiting_times.values()))
        overall_std = np.std(list(episode_mean_waiting_times.values()))
        min_waiting_time = min(episode_mean_waiting_times.values())
        min_episode = min(episode_mean_waiting_times, key=episode_mean_waiting_times.get)
        min_file_name = episode_files[min_episode]
        return episode_mean_waiting_times, overall_mean, overall_std, min_episode, min_waiting_time, min_file_name
    else:
        st.warning("No episodes were processed successfully.")
        return None, None, None, None, None, None


# Function to plot waiting time
def plot_waiting_time(file):
    if file is not None:
        if isinstance(file, pd.DataFrame):
            csv_df = file
            plt.title("Episode Waiting Time", fontsize=16)
        else:
            csv_df = pd.read_csv(file)
        overall_mean = csv_df["system_mean_waiting_time"].mean()
        plt.figure(figsize=(12, 6))
        plt.plot(csv_df.index, csv_df["system_mean_waiting_time"], label='Waiting Time', color='#2b8cbe')
        plt.title("Episode Waiting Time", fontsize=24)
        plt.axhline(y=overall_mean, color='r', linestyle='--', label=f'Overall Mean ({overall_mean:.2f})')
        plt.xlabel("Time Step", fontsize=14)
        plt.ylabel("Mean Waiting Time", fontsize=14)
        plt.grid(True)
        plt.legend()
        st.pyplot(plt)
    else:
        st.warning("File not provided.")


# Function to plot reward from JSON
def plot_reward_from_json(json_file, episode_num=0):
    if json_file is not None:
        stringio = StringIO(json_file.getvalue().decode("utf-8"))
        json_results = [json.loads(line) for line in stringio.readlines()]
        if json_results:
            result_grid = json_results[episode_num]
            values = result_grid["env_runners"]["hist_stats"]["episode_reward"]
            plt.figure(figsize=(12, 6))
            plt.plot(values, marker='o', linestyle='-', color='#31a354')
            plt.title("Reward Over Episodes", fontsize=24)
            plt.xlabel('Episode number', fontsize=14)
            plt.ylabel('Reward value', fontsize=14)
            plt.grid(True)
            st.pyplot(plt)
        else:
            st.warning("No data available in the JSON file.")
    else:
        st.warning("JSON file not provided.")


# Function to plot episode mean return
def plot_episode_mean_return(csv_file):
    if csv_file is not None:
        stringio = StringIO(csv_file.getvalue().decode("utf-8"))
        df = pd.read_csv(stringio)
        df = df.dropna(subset=['env_runners/episode_return_mean'])

        if 'env_runners/episode_return_mean' in df.columns:
            values = df['env_runners/episode_return_mean']
            plt.figure(figsize=(12, 6))
            plt.plot(values, marker='o', linestyle='-', color='#31a354')
            plt.title("Mean Return Over Episodes", fontsize=24)
            plt.xlabel('Episode number', fontsize=14)
            plt.ylabel("Mean Return", fontsize=14)
            plt.grid(True)
            st.pyplot(plt)
        else:
            st.warning("The column 'env_runners/episode_return_mean' is not available in the CSV file.")
    else:
        st.warning("CSV file not provided.")


# Function to generate PDF report with summary statistics at the beginning
def generate_pdf_report(min_file_name, progress_csv_file, json_file, episode_num, overall_mean, overall_std, min_waiting_time, min_episode, file_name):
    with PdfPages('report.pdf') as pdf:
        # First page: Summary Statistics
        plt.figure(figsize=(8, 6))
        plt.text(0.1, 0.9, f"Summary Statistics for {file_name}", fontsize=18, fontweight='bold')
        plt.text(0.1, 0.75, f"Overall Mean Waiting Time: {overall_mean:.3f}", fontsize=14)
        plt.text(0.1, 0.6, f"Overall Standard Deviation: {overall_std:.3f}", fontsize=14)
        plt.text(0.1, 0.45, f"Minimum Waiting Time: {min_waiting_time:.3f}", fontsize=14)
        plt.text(0.1, 0.3, f"Episode with Minimum Waiting Time: {min_episode}", fontsize=14)
        plt.axis('off')  # Turn off the axis
        pdf.savefig()  # Save the summary statistics as the first page of the PDF
        plt.close()  # Close the current figure

        # Second page: Mean Waiting Times for All Episodes
        plt.figure(figsize=(12, 6))
        if min_file_name is not None:
            csv_df = min_file_name if isinstance(min_file_name, pd.DataFrame) else pd.read_csv(min_file_name)
            overall_mean = csv_df["system_mean_waiting_time"].mean()
            plt.plot(csv_df.index, csv_df["system_mean_waiting_time"], label='Waiting Time', color='#2b8cbe')
            plt.axhline(y=overall_mean, color='r', linestyle='--', label=f'Overall Mean ({overall_mean:.2f})')
            plt.title("Episode With The Best Waiting Time", fontsize=24)
            plt.xlabel("Time Step", fontsize=14)
            plt.ylabel("Mean Waiting Time", fontsize=14)
            plt.grid(True)
            plt.legend()
            pdf.savefig()  # Save the current figure to the PDF
        plt.close()  # Close the current figure

        # Third page: Episode Mean Return
        plt.figure(figsize=(12, 6))
        if progress_csv_file is not None:
            stringio = StringIO(progress_csv_file.getvalue().decode("utf-8"))
            df = pd.read_csv(stringio)
            df = df.dropna(subset=['env_runners/episode_return_mean'])
            if 'env_runners/episode_return_mean' in df.columns:
                values = df['env_runners/episode_return_mean']
                plt.plot(values, marker='o', linestyle='-', color='#31a354')
                plt.title("Mean Return Over Episodes", fontsize=24)
                plt.xlabel('Episode number', fontsize=14)
                plt.ylabel("Mean Return", fontsize=14)
                plt.grid(True)
                pdf.savefig()  # Save the current figure to the PDF
        plt.close()  # Close the current figure

        # Fourth page: Reward Over Episodes from JSON
        plt.figure(figsize=(12, 6))
        if json_file is not None:
            stringio = StringIO(json_file.getvalue().decode("utf-8"))
            json_results = [json.loads(line) for line in stringio.readlines()]
            if json_results:
                result_grid = json_results[episode_num]
                values = result_grid["env_runners"]["hist_stats"]["episode_reward"]
                plt.plot(values, marker='o', linestyle='-', color='#31a354')
                plt.title("Reward Over Episodes", fontsize=24)
                plt.xlabel('Episode number', fontsize=14)
                plt.ylabel('Reward value', fontsize=14)
                plt.grid(True)
                pdf.savefig()  # Save the current figure to the PDF
        plt.close()  # Close the current figure


# Main function
def main():
    min_file_name = None
    set_custom_style()
    st.markdown("### ***Analysis and Visualizations for Traffic Tune Experiments***")
    st.markdown("> Upload your desired files")

    # Sidebar for file uploaders
    st.sidebar.header("Files for Analysis")
    csv_files = st.sidebar.file_uploader("Upload Episodes CSV Files", accept_multiple_files=True, type="csv", key="upload_files")
    progress_csv_file = st.sidebar.file_uploader("Upload Progress CSV File", type="csv", key="upload_csv_progress")
    json_file = st.sidebar.file_uploader("Upload Return JSON File", type="json", key="upload_json")
    episode_num = st.sidebar.number_input("Episode Index", min_value=-1, max_value=100, value=-1)

    # Analysis and plots
    st.markdown("#### Analysis and Plots")

    # Analyze episodes
    if csv_files:
        episode_mean_waiting_times, overall_mean, overall_std, min_episode, min_waiting_time, min_file_name = analyze_episodes(csv_files)
        if episode_mean_waiting_times is not None:
            st.subheader("Mean Waiting Times for All Episodes")
            with st.expander("View Mean Waiting Times"):
                episode_mean_waiting_times_with_numbers = {f"{k}": v for k, v in episode_mean_waiting_times.items()}
                st.write(episode_mean_waiting_times_with_numbers)
            st.subheader("Overall Metrics")
            st.metric("Overall Mean Waiting Time", f"{overall_mean:.3f}")
            st.metric("Overall Standard Deviation", f"{overall_std:.3f}")
            st.metric(f"Minimum Waiting Time (Episode {min_episode})", f"{min_waiting_time:.3f}")
            plot_waiting_time(min_file_name)
        else:
            st.warning("Please upload CSV files for analysis.")
    else:
        st.warning("* Please upload Episodes CSV files for analysis")

    # Plot return from CSV
    if progress_csv_file:
        # st.subheader("Mean Return Over Episodes")
        plot_episode_mean_return(progress_csv_file)
    else:
        st.warning("* Please upload a CSV progress file to plot returns")

    # Plot reward from JSON
    if json_file:
        # st.subheader("Reward Over Episodes")
        plot_reward_from_json(json_file, episode_num)
    else:
        st.warning("* Please upload a JSON file to plot rewards")

    file_name = st.sidebar.text_input("PDF Report Name")
    # Generate PDF report
    if st.sidebar.button("Generate PDF Report"):
        generate_pdf_report(min_file_name, progress_csv_file, json_file, episode_num, overall_mean, overall_std,
                            min_waiting_time, min_episode, file_name)
        st.success("PDF report generated successfully!")
        with open("report.pdf", "rb") as file:
            btn = st.download_button(
                label="Download PDF",
                data=file,
                file_name=file_name,
                mime="application/pdf"
            )
        st.balloons()


if __name__ == "__main__":
    main()
