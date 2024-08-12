import pandas as pd
import xml.etree.ElementTree as ET
import shutil as sh
import os


def add_vehicle_flows(original_xml_file_path: str, text_xml_file: str, inset_flows: list, hour: str) -> None:
    """
    Add vehicle flows to an XML file and save a new copy with flows.

    This function creates a new XML file with updated vehicle flows based on the original XML file.
    It removes existing flows and adds new ones based on the provided data.

    Parameters:
    - original_xml_file_path (str): Path to the original XML file.
    - text_xml_file (str): Path to the text file where new XML paths are logged.
    - inset_flows (list): List of dictionaries containing flow information to insert.
    - hour (str): Hour identifier for the new file name.

    Note:
    - The function assumes the existence of a global 'intersection_num' variable.
    - New XML files are named based on the hour and intersection number.
    - The function creates necessary directories if they don't exist.
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


def create_flow_files(data_frame):
    """
    Create flow files for each hour based on the provided DataFrame.

    This function iterates through each row of the DataFrame, where each row represents an hour,
    and creates flow files for that hour.

    Parameters:
    - data_frame (pd.DataFrame): DataFrame containing flow data for different hours and routes.
    - intersection (int): Intersection number.

    Note:
    - The function assumes the existence of global variables 'xml_rou_path' and 'text_xml_path_file'.
    - It skips columns named 'שעה' (hour) and any unnamed columns.
    - The column names are used as route identifiers.
    """
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


xlsx_file_path = 'flow_hour_all.xlsx'

abspath = os.path.dirname(os.path.abspath(__file__)).split('/Route_Generators')[0]

for i in range(1, 7):
    """
    Process flow data for intersections 1 through 6.

    This loop reads data from an Excel file for each intersection, creates necessary file paths,
    and generates flow files for each hour of data.

    Note:
    - The Excel file 'flow_hour_all.xlsx' is expected to have sheets named 'intersection_1' through 'intersection_6'.
    - XML and text files are created or overwritten in the specified paths.
    """
    intersection_num = i

    sheet_name = f'intersection_{intersection_num}'

    df = pd.read_excel(xlsx_file_path, sheet_name=sheet_name)

    xml_rou_path = (f"{abspath}/Nets/intersection_{intersection_num}"
                    f"/routes_{intersection_num}/intersection_{intersection_num}.rou.xml")

    print(xml_rou_path)
    text_xml_path_file = f'{abspath}/Nets/intersection_{intersection_num}/route_xml_path_intersection_{intersection_num}_hour.txt'
    print(text_xml_path_file)

    if os.path.exists(text_xml_path_file):
        os.remove(text_xml_path_file)
    create_flow_files(df)

print("Done!")

