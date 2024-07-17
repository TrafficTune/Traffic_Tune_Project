import pandas as pd
import xml.etree.ElementTree as ET
import shutil as sh
import os


def add_vehicle_flows(original_xml_file_path: str, text_xml_file: str, inset_flows: list, hour: str) -> None:
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
    new_xml_file_path = original_xml_file_path.replace('.rou.xml', f'_{hour}.rou.xml')
    new_xml_file_path = new_xml_file_path.replace(f'routes_{intersection_num}', f'routes_{intersection_num}_hour')

    # Ensure the directory exists
    new_dir = os.path.dirname(new_xml_file_path)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

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


def create_flow_files(data_frame, intersection):
    for index, row in data_frame.iterrows():
        hour = str(row['שעה'])  # Replace 'שעה' with the actual column name for hours
        hour = hour.replace(":", "_").replace("-", "").replace("00", "")
        flows = []
        new_index = 0
        for col in data_frame.columns:
            if col != 'שעה' and not col.startswith('Unnamed'):  # Skip the 'שעה' and unnamed columns
                vehs_per_hour = row[col]
                route = col  # Use column name as route
                flow = {
                    "id": f"f_{new_index}",
                    "begin": "0.00",
                    "route": route,
                    "end": "3600.00",
                    "vehsPerHour": f"{vehs_per_hour}",
                }
                flows.append(flow)
                new_index += 1

        add_vehicle_flows(xml_rou_path, text_xml_path_file, flows, hour)


# Function to create flow files for each row
intersection_num = 6

xlsx_file_path = f'/Users/eviat/Desktop/Final_Project/flow_count_tltan/intersection_{intersection_num}.xlsx'

sheet_name = 'a'
df = pd.read_excel(xlsx_file_path, sheet_name=sheet_name)

xml_rou_path = f"/Users/eviat/Desktop/Final_Project/Traffic_Tune_Project/Nets/intersection_{intersection_num}/routes_{intersection_num}/intersection_{intersection_num}.rou.xml"
text_xml_path_file = f'Nets/intersection_{intersection_num}/route_xml_path_intersection_{intersection_num}_hour.txt'
if os.path.exists(text_xml_path_file):
    os.remove(text_xml_path_file)
create_flow_files(df, intersection_num)

print("Done!")

