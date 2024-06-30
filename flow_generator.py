import xml.etree.ElementTree as ET
import random
import shutil as sh


def add_vehicle_flows(original_xml_file_path: str, text_xml_file: str, inset_flows: list, new_index: int, level: str):
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


def get_routes(xml_file: str):
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


def generate_new_flows(routes: list, low: int, high: int):
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


# Initialize variables and paths
num_of_intersection = "1"
xml_file_path = f'Nets/intersection_{num_of_intersection}/routes_{num_of_intersection}/intersection_{num_of_intersection}.rou.xml'
text_xml_path_file = f'Nets/intersection_{num_of_intersection}/route_xml_path_intersection_{num_of_intersection}.txt'

# Initialize lists and index
new_flows = []
xml_new_index = 1

# Get routes from XML file
routes_list = get_routes(xml_file_path)

# Generate and update flows for 100 files with varying difficulty levels
for i in range(100):
    if i >= 75:
        level_flow = "very_hard"
        from_flow = 1800
        to_flow = 2200
    elif i >= 50:
        level_flow = "hard"
        from_flow = 1000
        to_flow = 1800
    elif i >= 25:
        level_flow = "medium"
        from_flow = 500
        to_flow = 1000
    else:
        level_flow = "easy"
        from_flow = 200
        to_flow = 500

    # Generate new flows
    new_flows = generate_new_flows(routes_list, from_flow, to_flow)

    # add vehicle flows and log new XML file path
    add_vehicle_flows(xml_file_path, text_xml_path_file, new_flows, xml_new_index, level_flow)

    # Increment index for next file
    xml_new_index += 1

print("Done")
