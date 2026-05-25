import os
import pandas as pd
from glob import glob

# -------------------------------
# Define paths using BASE_PATH
# Without using base path we can get path issues when running from different locations
# -------------------------------
# BASE_PATH = project root folder (go up 3 levels from src/ingestion/log_ingestion.py)
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RAW_PATH = os.path.join(BASE_PATH, "data", "raw")
PROCESSED_PATH = os.path.join(BASE_PATH, "data", "processed")
OUTPUT_FILE = os.path.join(PROCESSED_PATH, "clean_logs.csv")

# Make sure processed folder exists
os.makedirs(PROCESSED_PATH, exist_ok=True)

# -------------------------------
# Find all CSV files in raw folder
# -------------------------------
csv_files = glob(os.path.join(RAW_PATH, "*.csv"))

# -------------------------------
# Read and process all CSV files
# -------------------------------
dfs = []

for file in csv_files:
    df = pd.read_csv(file)

    # -------------------------------
    # Normalize Sysmon columns to our pipeline format
    # -------------------------------
    col_map = {
        'EventID': 'event_id',
        'Computer': 'host',
        'SourceIp': 'source_ip',
        'DestinationPortName': 'dest_port',
        'DestinationIp': 'dest_ip',
        'SourcePort': 'source_port',
        'Image': 'process_name',
        'CommandLine': 'command_line',
        'User': 'user',
        'UtcTime': 'timestamp'
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    print("Columns after rename:", df.columns)

    # Ensure timestamp exists and is datetime
    if 'timestamp' not in df.columns:
        import datetime
        start_time = datetime.datetime(2025, 11, 15, 8, 0, 0)
        df['timestamp'] = [start_time + datetime.timedelta(minutes=i) for i in range(len(df))]
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, errors='coerce')

    # Fill in required columns if missing
    if 'user' not in df.columns:
        df['user'] = 'sysmon_user'
    if 'host' not in df.columns:
        df['host'] = 'UNKNOWN_HOST'
    if 'event_id' not in df.columns:
        df['event_id'] = 0
    if 'process_name' not in df.columns:
        df['process_name'] = '-'
    if 'command_line' not in df.columns:
        df['command_line'] = '-'
    if 'source_ip' not in df.columns:
        df['source_ip'] = '-'

    # Keep only required columns
    columns_to_keep = ['timestamp', 'user', 'host', 'event_id', 'process_name', 'command_line', 'source_ip']
    for extra_col in ['dest_ip', 'source_port', 'dest_port']:
        if extra_col in df.columns:
            columns_to_keep.append(extra_col)
    
    df = df[[c for c in columns_to_keep if c in df.columns]]

    dfs.append(df)

# Concatenate all dataframes
if dfs:
    all_logs = pd.concat(dfs, ignore_index=True)
else:
    all_logs = pd.DataFrame(columns=['timestamp', 'user', 'host', 'event_id', 'process_name', 'command_line', 'source_ip', 'dest_ip', 'source_port', 'dest_port'])

# Sort by timestamp
all_logs = all_logs.sort_values('timestamp')

# -------------------------------
# Save processed logs
# -------------------------------
all_logs.to_csv(OUTPUT_FILE, index=False)
print(f"Processed logs saved to {OUTPUT_FILE}")
