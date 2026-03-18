import os
import glob
import re
import pandas as pd

# ==========================================
# 1. Environment Mapping
# ==========================================
LOCATION_MAPPING = {
    'garden': 0,
    'forest': 1,
    'lake': 2,
    'river': 3,
    'bridge': 4
}

def extract_node_id(name):
    """Extracts the numerical node ID from the folder name (e.g., 'node_2' -> 2)."""
    match = re.search(r'\d+', name)
    return int(match.group()) if match else None

def get_env_id(folder_name):
    """Finds the matching Environment ID based on the folder name string."""
    name_lower = folder_name.lower()
    for loc_name, env_id in LOCATION_MAPPING.items():
        if loc_name in name_lower:
            return env_id
    return None

def parse_txt_file(filepath, folder_node_id, env_id):
    """Parses a single RIOT log file and tags it with the folder IDs."""
    parsed_data = []
    
    with open(filepath, 'r', errors='replace') as file:
        for line in file:
            if '# [DATA]' in line:
                try:
                    _, data_part = line.split('# [DATA]')
                    parts = data_part.strip().split(',')
                    
                    tx_node_id = int(parts[0].strip()) 
                    timestamp = int(parts[1].strip())
                    rssi = int(parts[2].strip())
                    
                    parsed_data.append({
                        'receiver_id': folder_node_id, 
                        'env_id': env_id,          
                        'sender_id': tx_node_id,  
                        'timestamp': timestamp,
                        'rssi': rssi
                    })
                except Exception:
                    continue
                    
    return parsed_data

def main():
    # 1. Get the absolute path to the directory where this script lives
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Define paths strictly relative to the script's location
    data_dir = os.path.join(script_dir, '../data')
    output_csv = os.path.join(script_dir, '../data/dataset.csv')
    
    all_rows = []
    
    print(f"Scanning directory: {os.path.abspath(data_dir)} for Node and Location folders...\n")

    if not os.path.exists(data_dir):
        print(f"ERROR: The folder {os.path.abspath(data_dir)} does not exist.")
        return

    # 3. Iterate through Node folders
    for node_folder in os.listdir(data_dir):
        node_path = os.path.join(data_dir, node_folder)
        
        if not os.path.isdir(node_path):
            continue
            
        folder_node_id = extract_node_id(node_folder)
        if folder_node_id is None:
            continue
            
        # 4. Iterate through Location folders 
        for loc_folder in os.listdir(node_path):
            loc_path = os.path.join(node_path, loc_folder)
            
            if not os.path.isdir(loc_path):
                continue
                
            env_id = get_env_id(loc_folder)
            if env_id is None:
                continue 
                
            # 5. Find all .txt files
            txt_files = glob.glob(os.path.join(loc_path, '*.txt'))
            
            for txt_file in txt_files:
                file_data = parse_txt_file(txt_file, folder_node_id, env_id)
                all_rows.extend(file_data)
                print(f"Parsed {os.path.basename(txt_file):<25} | Node: {folder_node_id} | Env: {env_id} ({loc_folder:<6}) | Entries: {len(file_data)}")

    # 6. Combine and Save
    if all_rows:
        final_df = pd.DataFrame(all_rows)
        
        # Sort chronologically
        final_df.sort_values(by=['sender_id', 'env_id'], inplace=True)
        
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        final_df.to_csv(output_csv, index=False)
        print(f"\n✅ Success! Saved {len(final_df)} total measurements to {os.path.abspath(output_csv)}")
    else:
        print("\n❌ No data was successfully parsed. Please check your folder structure and file names.")

if __name__ == "__main__":
    main()
