import os
import pandas as pd

# -----------------------
# Paths
# -----------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ALERT_DIR = os.path.join(BASE_DIR, "data", "alerts")
ML_ALERTS_FILE = os.path.join(ALERT_DIR, "ml_alerts.csv")
RULE_ALERTS_FILE = os.path.join(ALERT_DIR, "alerts.csv")
ALL_ALERTS_FILE = os.path.join(ALERT_DIR, "all_alerts.csv")

print("[+] Merging alerts...")

# -----------------------
# Load ML alerts
# -----------------------
if os.path.exists(ML_ALERTS_FILE):
    ml_alerts = pd.read_csv(ML_ALERTS_FILE)
    # Ensure time_window is datetime for sorting
    if 'time_window' in ml_alerts.columns:
        ml_alerts['time_window'] = pd.to_datetime(ml_alerts['time_window'])
    print(f"[+] Loaded {len(ml_alerts)} ML alerts")
else:
    print(f"[!] ML alerts file not found: {ML_ALERTS_FILE}")
    ml_alerts = pd.DataFrame()

# -----------------------
# Load rule-based alerts
# -----------------------
if os.path.exists(RULE_ALERTS_FILE):
    rule_alerts = pd.read_csv(RULE_ALERTS_FILE)
    # Convert timestamp to time_window for consistency
    if 'timestamp' in rule_alerts.columns:
        rule_alerts['timestamp'] = pd.to_datetime(rule_alerts['timestamp'])
        rule_alerts['time_window'] = rule_alerts['timestamp'].dt.floor("10min")
    print(f"[+] Loaded {len(rule_alerts)} rule-based alerts")
else:
    print(f"[!] Rule-based alerts file not found: {RULE_ALERTS_FILE}")
    rule_alerts = pd.DataFrame()

# -----------------------
# Standardize and merge alerts
# -----------------------
all_alerts_list = []

# Process ML alerts
if not ml_alerts.empty:
    for _, row in ml_alerts.iterrows():
        all_alerts_list.append({
            'time_window': row.get('time_window', ''),
            'timestamp': row.get('time_window', ''),  # Use time_window as timestamp
            'user': row.get('user', ''),
            'host': row.get('host', ''),
            'source_ip': row.get('source_ip', ''),
            'dest_ip': '',
            'source_port': '',
            'dest_port': '',
            'process_name': row.get('process_name', ''),
            'alert_type': row.get('alert_type', 'ML anomaly'),
            'details': row.get('details', ''),
            'source': 'ML'
        })

# Process rule-based alerts
if not rule_alerts.empty:
    for _, row in rule_alerts.iterrows():
        all_alerts_list.append({
            'time_window': row.get('time_window', row.get('timestamp', '')),
            'timestamp': row.get('timestamp', ''),
            'user': row.get('user', ''),
            'host': row.get('host', ''),
            'source_ip': row.get('source_ip', ''),
            'dest_ip': row.get('dest_ip', ''),
            'source_port': row.get('source_port', ''),
            'dest_port': row.get('dest_port', ''),
            'process_name': row.get('process_name', ''),
            'alert_type': row.get('alert_type', ''),
            'details': row.get('details', ''),
            'source': 'Rule-based'
        })

# -----------------------
# True Hybrid Correlation Logic
# -----------------------
# Group alerts by (user, time_window) to find intersections
from collections import defaultdict

user_time_groups = defaultdict(set)
for idx, alert in enumerate(all_alerts_list):
    user = alert['user']
    time_win = alert['time_window']
    source = alert['source']
    
    if user and time_win:
        user_time_groups[(user, time_win)].add(source)

# Upgrade correlated alerts to 'Hybrid'
hybrid_count = 0
for alert in all_alerts_list:
    user = alert['user']
    time_win = alert['time_window']
    
    if user and time_win:
        sources_in_window = user_time_groups[(user, time_win)]
        if 'ML' in sources_in_window and 'Rule-based' in sources_in_window:
            alert['source'] = 'Hybrid'
            hybrid_count += 1

print(f"[+] Upgraded {hybrid_count} alerts to Hybrid status based on cross-engine correlation.")

# -----------------------
# Create merged DataFrame and sort
# -----------------------
if all_alerts_list:
    all_alerts_df = pd.DataFrame(all_alerts_list)
    
    # Convert time_window to datetime for sorting
    all_alerts_df['time_window'] = pd.to_datetime(all_alerts_df['time_window'])
    all_alerts_df['timestamp'] = pd.to_datetime(all_alerts_df['timestamp'])
    
    # Sort by time_window
    all_alerts_df = all_alerts_df.sort_values('time_window')
    
    # Save merged alerts (with error handling for locked files)
    try:
        all_alerts_df.to_csv(ALL_ALERTS_FILE, index=False)
        print(f"[OK] Merged alerts saved to {ALL_ALERTS_FILE}")
    except (PermissionError, OSError) as e:
        # If file is locked, write to a backup file
        import datetime
        backup_file = ALL_ALERTS_FILE.replace('.csv', f'_backup_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        try:
            all_alerts_df.to_csv(backup_file, index=False)
            print(f"[!] Warning: Could not write to {ALL_ALERTS_FILE} (file may be open in another program)")
            print(f"[OK] Merged alerts saved to backup file: {backup_file}")
            print("[!] Please close the file and run again, or use the backup file")
        except Exception as e2:
            print(f"[ERROR] Could not save merged alerts: {e2}")
            raise
    print(f"[OK] Total alerts: {len(all_alerts_df)}")
    print(f"[OK]   - ML alerts: {len(ml_alerts)}")
    print(f"[OK]   - Rule-based alerts: {len(rule_alerts)}")
    
    # Print summary by alert type
    print("\n[+] Alert summary by type:")
    print(all_alerts_df['alert_type'].value_counts())
    
    # Print summary by user
    print("\n[+] Alert summary by user:")
    print(all_alerts_df['user'].value_counts())
else:
    print("[!] No alerts to merge!")

