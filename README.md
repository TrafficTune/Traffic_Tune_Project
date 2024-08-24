# Traffic Tune Project
![WhatsApp Image 2024-06-27 at 20 26 43](https://github.com/TrafficTune/Traffic_Tune_Project/assets/73496652/c71694bb-5653-4a4e-b20a-05a5c9d263a5)

## Overview
The Traffic Tune project aims to apply reinforcement learning (RL) to optimize traffic signal control in urban environments, focusing on improving traffic flow and reducing congestion. By leveraging advanced RL algorithms and simulation tools, this project seeks to develop a dynamic and adaptive traffic management system that adjusts traffic light timings based on real-time conditions.

The project utilizes the SUMO (Simulation of Urban MObility) platform for traffic simulation, and Ray RLLib for training and deploying RL agents. Various RL algorithms including Deep Q-Network (DQN), Double Dueling DQN (DDQN), Proximal Policy Optimization (PPO), and Asynchronous Proximal Policy Optimization (APPO) were implemented and tested.

## Project Structure
The project is organized into several modules, each handling different components of the system:

- **Configurations**: Defines the environment and algorithm configurations.
- **Simulation Environment Manager**: Manages the SUMO simulation and integrates real-world traffic data.
- **Route Generators**: Generates realistic traffic routes for simulation.
- **Algorithm Trainer**: Responsible for training and evaluation of the RL models using Ray RLLib.
- **Utilities**: Contains helper functions for data preprocessing, reward function settings, etc.
- **Visualizations and Metrics**: Provides tools for analyzing and visualizing the results of the simulations.
- **Networks**: Defines the simulation networks.
- **Outputs**: Stores the results and model outputs from training sessions.

## Algorithms Used
This project implements and experiments with several RL algorithms, including:
- **Deep Q-Network (DQN)**
- **Double Dueling DQN (DDQN)**
- **Proximal Policy Optimization (PPO)**
- **Asynchronous Proximal Policy Optimization (APPO)**

## Installation

### Step 1: Clone the repository
```bash
git clone https://github.com/TrafficTune/Traffic_Tune_Project.git
```

### Step 2: Install SUMO
To run the traffic simulation, you will need to install SUMO. Please refer to the official SUMO installation documentation for detailed instructions based on your operating system:

[SUMO Installation Guide](https://sumo.dlr.de/docs/Downloads.php)

### Step 3: Install Python dependencies
Navigate to the project directory and install the necessary Python dependencies:
```bash
pip install -r requirements.txt
```

## Running the Simulation

To run the simulation, follow the steps outlined in the `main_notebook.ipynb`. Open the notebook and execute the cells in sequence to initialize the SUMO environment, train the RL models, and evaluate performance.

### Step 1: Open Jupyter Notebook
Start Jupyter Notebook by running:
```bash
jupyter notebook
```

### Step 2: Run the `main_notebook.ipynb`
In the Jupyter interface, open `main_notebook.ipynb` and follow the step-by-step instructions in the notebook to run the simulation and training process.

## Usage
The project provides several configurable parameters for customizing the simulations, such as:
- Number of intersections
- Traffic routes
- Algorithm selection (DQN, DDQN, PPO, APPO)
- Hyper-parameter settings

You can adjust these parameters in the configuration files located in the `Config/` directory.

## Results
The results of the experiments demonstrate that RL-based traffic signal control significantly reduces vehicle waiting times and improves traffic flow compared to traditional fixed-time signal systems.

## Citation
If you use this repository for your research or projects, please cite:
```bibtex
@misc{traffictune2024,
  author = {Tal Mekler and Eviatar Didon and Daniel Margolin and Matan Drabkin},
  title = {Traffic Tune Project Repository},
  year = {2024},
  note = {Available at: \url{https://github.com/TrafficTune/Traffic_Tune_Project}. Accessed: 2024-08-22}
}
```

## Contributors
- **Tal Mekler** 
- **Eviatar Didon**
- **Daniel Margolin**
- **Matan Drabkin**

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
