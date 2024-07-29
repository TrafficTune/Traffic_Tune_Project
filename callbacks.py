import traci
import logging
from ray.rllib.algorithms.callbacks import DefaultCallbacks

# Configure logging
logging.basicConfig(filename='waiting_times.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class AverageWaitingTimeCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.waiting_times = {}  # Dictionary to store waiting times and vehicle counts for each agent
        self.episode_count = 1

    def on_episode_start(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        # Initialize waiting times and vehicle count for each agent at the start of the episode
        try:
            # Try to get the actual environment from the base_env
            env = base_env.get_sub_environments()[env_index]

            # Check if the environment has a get_agent_ids method
            if hasattr(env, 'get_agent_ids'):
                agent_ids = env.get_agent_ids()
                logging.info(f"Agent IDs(multi_agent): {agent_ids}")
            elif hasattr(env, 'ts_ids'):
                agent_ids = env.ts_ids
                logging.info(f"Agent IDs(ts_ids): {agent_ids}")
            else:
                # Fallback to using policy keys if neither method is available
                agent_ids = list(policies.keys())
                logging.info(f"Using policy keys: {agent_ids}")
        except Exception as e:
            # If any error occurs, fallback to using policy keys
            agent_ids = list(policies.keys())
            logging.info(f"Exception occurred: {str(e)}. Using policy keys: {agent_ids}")

        self.waiting_times = {agent_id: {'total_waiting_time': 0, 'vehicle_count': 0}
                              for agent_id in agent_ids}

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        """
        Collect waiting time data at the end of each episode for each agent.
        """
        for agent_id in self.waiting_times.keys():
            episode_waiting_time = 0
            vehicle_count = 0

            # Iterate through all vehicles for the current agent
            for vehicle_id in traci.vehicle.getIDList():
                vehicle_lane_id = traci.vehicle.getLaneID(vehicle_id)

                # Filter vehicles for the agent's area
                # TODO: needs to be fixed. in single agent case, lane_id has no agent_id in it.
                #  also, in multi-agent case - but then we dont know if the lane_id is the agent's area

                if traci.trafficlight.getControlledLanes(agent_id) == vehicle_lane_id:
                    waiting_time = traci.vehicle.getWaitingTime(vehicle_id)
                    episode_waiting_time += waiting_time
                    vehicle_count += 1

            # Update the agent's waiting time and vehicle count
            episode.custom_metrics[f"{agent_id}_total_waiting_time"] = episode_waiting_time
            episode.custom_metrics[f"{agent_id}_vehicle_count"] = vehicle_count

            # Log the total waiting time for the agent in this episode
            episode.custom_metrics[f"{agent_id}_episode_total_waiting_time"] = episode_waiting_time
            self.episode_count += 1

    def on_train_result(self, *, algorithm, result, **kwargs):
        """
        Calculate the average waiting time for each agent after training using RLlib's custom metrics.
        """

        waiting_times = {}
        mean_waiting_times = {}
        # Extract waiting times and vehicle counts from custom metrics
        for key, value in result['env_runners']["custom_metrics"].items():
            if "episode_total_waiting_time" in key:
                agent_id = key.replace("_episode_total_waiting_time", "")
                if agent_id not in waiting_times:
                    waiting_times[agent_id] = {"total_waiting_time": 0, "vehicle_count": 0}
                waiting_times[agent_id]["total_waiting_time"] += value
            elif "vehicle_count" in key:
                agent_id = key.replace("_vehicle_count", "")
                if agent_id not in waiting_times:
                    waiting_times[agent_id] = {"total_waiting_time": 0, "vehicle_count": 0}
                waiting_times[agent_id]["vehicle_count"] += value

        # Calculate mean waiting times
        for agent_id, data in waiting_times.items():
            total_waiting_time = data['total_waiting_time']
            vehicle_count = data['vehicle_count']

            if vehicle_count > 0:
                mean_waiting_time = total_waiting_time / vehicle_count
            else:
                mean_waiting_time = 0

            mean_waiting_times[agent_id] = mean_waiting_time
            logging.info(f"Training Result - Agent {agent_id}: {mean_waiting_time} seconds")

            # Store the mean waiting time in the result for each agent
            result["custom_metrics"][f"{agent_id}_mean_waiting_time"] = mean_waiting_time

        # Log the final results
        logging.info(f"Training Results - Mean waiting times for all agents: {mean_waiting_times}")
