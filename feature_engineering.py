import pandas as pd
import os

# Correct paths based on your project structure
INPUT_CSV = "data/processed/clean_logs.csv"
OUTPUT_CSV = "data/features/feature_table.csv"

# Ensure output folder exists
os.makedirs("data/features", exist_ok=True)

print("[+] Loading cleaned logs...")

df = pd.read_csv(INPUT_CSV)

# Convert timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'])

print("[+] Generating features...")

# ---- Feature 1: Actions per user per 10-minute window ----
df['time_window'] = df['timestamp'].dt.floor("10min")
action_counts = (
    df.groupby(['user', 'time_window'])
      .size()
      .reset_index(name='action_count')
)

# ---- Feature 2: Number of unique hosts touched per window ----
unique_hosts = (
    df.groupby(['user', 'time_window'])['host']
      .nunique()
      .reset_index(name="unique_hosts")
)

# ---- Feature 3: Suspicious process execution count ----
df['is_suspicious_process'] = df['process_name'].isin([
    "powershell.exe", "cmd.exe", "wmic.exe", "psexec.exe"
]).astype(int)

suspicious_counts = (
    df.groupby(['user', 'time_window'])['is_suspicious_process']
      .sum()
      .reset_index(name="suspicious_process_count")
)

# ---- Feature 4: Number of unique source IPs per window ----
unique_source_ips = (
    df.groupby(['user', 'time_window'])['source_ip']
      .nunique()
      .reset_index(name="num_source_ips")
)

# ---- Collect all hosts and IPs used in the window ----
hosts_list = (
    df.groupby(['user', 'time_window'])['host']
      .unique()
      .apply(lambda x: ', '.join(set([str(h) for h in x if str(h) != '-'])))
      .reset_index(name="host_list")
)
ips_list = (
    df.groupby(['user', 'time_window'])['source_ip']
      .unique()
      .apply(lambda x: ', '.join(set([str(ip) for ip in x if str(ip) != '-'])))
      .reset_index(name="source_ip_list")
)

# ---- Feature 5: Number of unique commands per window ----
# Filter out rows where command_line is missing or just "-"
df_commands = df[df['command_line'].notna() & (df['command_line'] != '-')].copy()
unique_commands = (
    df_commands.groupby(['user', 'time_window'])['command_line']
      .nunique()
      .reset_index(name="num_unique_commands")
)

# ---- Feature 6: Outside work hours indicator ----
# Define work hours as 06:00-18:00, so outside is 00:00-06:00 or 18:00-23:59
df['hour'] = df['timestamp'].dt.hour
df['outside_work_hours'] = ((df['hour'] >= 0) & (df['hour'] < 6)) | (df['hour'] >= 18)
outside_work_hours = (
    df.groupby(['user', 'time_window'])['outside_work_hours']
      .any()  # True if any activity in window is outside work hours
      .astype(int)
      .reset_index(name="outside_work_hours")
)

# ---- Combine all features ----
feature_table = action_counts.merge(unique_hosts, on=['user', 'time_window'])
feature_table = feature_table.merge(suspicious_counts, on=['user', 'time_window'])
feature_table = feature_table.merge(unique_source_ips, on=['user', 'time_window'], how='left')
feature_table = feature_table.merge(hosts_list, on=['user', 'time_window'], how='left')
feature_table = feature_table.merge(ips_list, on=['user', 'time_window'], how='left')
feature_table = feature_table.merge(unique_commands, on=['user', 'time_window'], how='left')
feature_table = feature_table.merge(outside_work_hours, on=['user', 'time_window'], how='left')

# Fill NaN values (for windows with no commands or source IPs)
feature_table['num_source_ips'] = feature_table['num_source_ips'].fillna(0).astype(int)
feature_table['num_unique_commands'] = feature_table['num_unique_commands'].fillna(0).astype(int)
feature_table['outside_work_hours'] = feature_table['outside_work_hours'].fillna(0).astype(int)
feature_table['host_list'] = feature_table['host_list'].fillna('')
feature_table['source_ip_list'] = feature_table['source_ip_list'].fillna('')

# ---- Feature 7: Structural Anomaly Scores (from Pre-trained Model) ----
import glob
import joblib

raw_files = glob.glob(os.path.join("data", "raw", "*.csv"))
if raw_files:
    raw_df = pd.concat([pd.read_csv(f) for f in raw_files], ignore_index=True)
    model_path = os.path.join("src", "detection", "isolation_forest_lmd2023_tuned.joblib")
    if os.path.exists(model_path):
        print("[+] Fusing structural anomalies from pre-trained model into behavioral features...")
        model_dict = joblib.load(model_path)
        model = model_dict['model']
        preprocessor = model_dict.get('preprocessor')
        expected_features = model_dict.get('features', [])
        
        for col in expected_features:
            if col not in raw_df.columns:
                if col in ['EventRecordID', 'Execution_ProcessID', 'ProcessId']:
                    raw_df[col] = 0
                else:
                    raw_df[col] = "Unknown"
                    
        X_raw = raw_df[expected_features]
        X_processed = preprocessor.transform(X_raw) if preprocessor else X_raw.values
        
        scores = -model.decision_function(X_processed)
        threshold = model_dict.get('threshold', 0.11)
        
        raw_df['structural_anomaly_score'] = scores
        raw_df['is_structural_anomaly'] = (scores > threshold).astype(int)
        
        # Ensure timestamp is parsed properly
        if 'UtcTime' in raw_df.columns:
            raw_df['timestamp'] = pd.to_datetime(raw_df['UtcTime'], dayfirst=True, errors='coerce')
        else:
            import datetime
            start_time = datetime.datetime(2025, 11, 15, 8, 0, 0)
            raw_df['timestamp'] = [start_time + datetime.timedelta(minutes=i) for i in range(len(raw_df))]
            
        raw_df['time_window'] = raw_df['timestamp'].dt.floor("10min")
        raw_df['user'] = raw_df.get('User', pd.Series(["sysmon_user"]*len(raw_df))).fillna("sysmon_user")
        
        structural_features = raw_df.groupby(['user', 'time_window']).agg(
            max_structural_anomaly_score=('structural_anomaly_score', 'max'),
            num_structural_anomalies=('is_structural_anomaly', 'sum')
        ).reset_index()
        
        feature_table = feature_table.merge(structural_features, on=['user', 'time_window'], how='left')
        feature_table['max_structural_anomaly_score'] = feature_table['max_structural_anomaly_score'].fillna(0)
        feature_table['num_structural_anomalies'] = feature_table['num_structural_anomalies'].fillna(0).astype(int)
    else:
        feature_table['max_structural_anomaly_score'] = 0
        feature_table['num_structural_anomalies'] = 0
else:
    feature_table['max_structural_anomaly_score'] = 0
    feature_table['num_structural_anomalies'] = 0

print("[+] Saving feature table to:", OUTPUT_CSV)

# Try to save, with error handling for locked files
try:
    feature_table.to_csv(OUTPUT_CSV, index=False)
    print("[OK] Feature engineering complete!")
    print("[OK] Output generated at data/features/feature_table.csv")
except (PermissionError, OSError) as e:
    # If file is locked, write to a backup file
    import datetime
    backup_file = OUTPUT_CSV.replace('.csv', f'_backup_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    try:
        feature_table.to_csv(backup_file, index=False)
        print(f"[!] Warning: Could not write to {OUTPUT_CSV} (file may be open in another program)")
        print(f"[OK] Feature table saved to backup file: {backup_file}")
        print("[!] Please close the file and run again, or use the backup file")
    except Exception as e2:
        print(f"[ERROR] Could not save feature table: {e2}")
        raise
