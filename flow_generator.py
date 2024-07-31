import xml.etree.ElementTree as ET
import random
import shutil as sh
import os


def add_vehicle_flows(original_xml_file_path: str, text_xml_file: str, inset_flows: list, new_index: int,
                      level: str) -> None:
    """
    Add vehicle flows in an XML file and save a new copy with flows.

    Parameters:
    - original_xml_file_path (str): Path to the original XML file.
    - text_xml_file (str): Path to the text file where new XML paths are logged.
    - inset_flows (list): List of dictionaries containing flow information to insert.
    - new_index (int): Index to append to the new XML file name.
    - level (str): Difficulty level of the flows (e.g., easy, medium, hard, very_hard).
    """
    # Construct new file path based on level and index
    new_xml_file_path = original_xml_file_path.replace('.rou.xml', f'_{level}_{new_index}.rou.xml')

    # Copy original XML file to new path
    sh.copyfile(original_xml_file_path, new_xml_file_path)

    # Parse the XML file
    tree = ET.parse(new_xml_file_path)
    root = tree.getroot()

    # Remove existing flow elements
    flows = root.findall('flow')
    for flow in flows:
        root.remove(flow)

    # Add new flow elements
    for flow_data in inset_flows:
        flow_element = ET.Element('flow')
        for key, value in flow_data.items():
            flow_element.set(key, value)
        root.append(flow_element)
        root.append(ET.Comment("\n"))

    # Write the updated XML tree back to the file
    tree.write(new_xml_file_path, encoding='unicode')

    # Append the new XML file path to the text file
    with open(text_xml_file, 'a') as f:
        f.write(f'{new_xml_file_path}\n')


def get_routes(xml_file: str) -> list:
    """
    Retrieve route IDs from an XML file.

    Parameters:
    - xml_file (str): Path to the XML file containing route information.

    Returns:
    - routes_list (list): List of route IDs extracted from the XML.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    routes_list = []
    for route in root.findall('route'):
        routes_list.append(route.get('id'))
    return routes_list


def generate_new_flows(routes: list, low: int, high: int) -> list:
    """
    Generate new flow data for routes.

    Parameters:
    - routes (list): List of route IDs.
    - low (int): Lower bound for random number generation.
    - high (int): Upper bound for random number generation.

    Returns:
    - new_flows (list): List of dictionaries containing generated flow data.
    """
    flows = []
    for i, route_id in enumerate(routes):
        vehs_per_hour = random.randint(low, high)
        flow = {
            "id": f"f_{i}",
            "begin": "0.00",
            "route": route_id,
            "end": "3600.00",
            "vehsPerHour": f"{vehs_per_hour}",
        }
        flows.append(flow)
    return flows


def random_flows(routes: list, high: float) -> list:
    """
    Generate new flow data for routes.

    Parameters:
    - routes (list): List of route IDs.
    - low (int): Lower bound for random number generation.
    - high (int): Upper bound for random number generation.

    Returns:
    - new_flows (list): List of dictionaries containing generated flow data.
    """
    flows = []
    for i, route_id in enumerate(routes):
        probability = str(random.uniform(0.1, high))
        flow = {
            "id": f"f_{i}",
            "begin": "0.00",
            "route": route_id,
            "end": "3600",
            "probability": probability,
            "departSpeed": "random",
            "departPos": "random",
            "departLane": "random",
        }
        flows.append(flow)
    return flows


# Generate and update flows for 100 files with varying difficulty levels
for j in range(1, 8):
    # Initialize variables and paths
    num_of_intersection = j
    xml_file_path = f'Nets/intersection_{num_of_intersection}/routes_{num_of_intersection}/intersection_{num_of_intersection}.rou.xml'
    text_xml_path_file = f'Nets/intersection_{num_of_intersection}/route_xml_path_intersection_{num_of_intersection}.txt'
    if os.path.exists(text_xml_path_file):
        os.remove(text_xml_path_file)

    # Initialize lists and index
    new_flows = []

    # Get routes from XML file
    routes_list = get_routes(xml_file_path)

    levels_random = {
        1: ("easy", 0.3),
        2: ("medium", 0.5),
        3: ("hard", 0.7),
        4: ("very_hard", 0.9),
    }
    for k in range(1, 5):
        level_flow, high_flow_probability = levels_random[k]
        # Generate random flows
        random_flow = random_flows(routes_list, high_flow_probability)
        add_vehicle_flows(xml_file_path, text_xml_path_file, random_flow, k, f"random_{level_flow}")

    level_difficulty = [
        (13, "very_hard", 3000, 4000),
        (9, "hard", 1500, 3000),
        (5, "medium", 750, 1500),
        (0, "easy", 500, 750),
    ]

    for i in range(1, 17):
        for boundary, level, from_flow, to_flow in level_difficulty:
            if i >= boundary:
                level_flow = level
                from_flow_value = from_flow
                to_flow_value = to_flow
                break

        # Generate new flows
        new_flows = generate_new_flows(routes_list, from_flow_value, to_flow_value)

        # Add vehicle flows and log new XML file path
        add_vehicle_flows(xml_file_path, text_xml_path_file, new_flows, i, level_flow)


print("Done")
