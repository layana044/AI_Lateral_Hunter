import os
import pandas as pd
from datetime import timedelta, time

# -------------------------------
# Paths setup
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLEAN_LOGS_PATH = os.path.join(BASE_DIR, "data", "processed", "clean_logs.csv")
ALERTS_PATH = os.path.join(BASE_DIR, "data", "alerts")
OUTPUT_FILE = os.path.join(ALERTS_PATH, "alerts.csv")
os.makedirs(ALERTS_PATH, exist_ok=True)

# -------------------------------
# Load cleaned logs
# -------------------------------
logs = pd.read_csv(CLEAN_LOGS_PATH)
logs['timestamp'] = pd.to_datetime(logs['timestamp'])

alerts = []

# -------------------------------
# Rule 1: Multiple hosts in short time (10 min)
# -------------------------------
for user, user_logs in logs.groupby('user'):
    user_logs = user_logs.sort_values('timestamp')
    recent_hosts = []  # list of tuples (host, timestamp)

    for i in range(len(user_logs)):
        current_time = user_logs.iloc[i]['timestamp']
        current_host = user_logs.iloc[i]['host']
        current_ip = user_logs.iloc[i].get('source_ip', '-')
        current_dest_ip = user_logs.iloc[i].get('dest_ip', '-')
        current_src_port = user_logs.iloc[i].get('source_port', '-')
        current_dst_port = user_logs.iloc[i].get('dest_port', '-')

        # Remove hosts older than 10 minutes
        recent_hosts = [(h, t) for h, t in recent_hosts if t >= current_time - timedelta(minutes=10)]

        # Check if any previous host is different
        previous_hosts = set(h for h, t in recent_hosts)
        if previous_hosts and current_host not in previous_hosts:
            alerts.append({
                'timestamp': current_time,
                'user': user,
                'host': current_host,
                'source_ip': current_ip,
                'dest_ip': current_dest_ip,
                'source_port': current_src_port,
                'dest_port': current_dst_port,
                'alert_type': 'Multiple hosts in short time',
                'details': f"User accessed hosts {previous_hosts} within 10 minutes"
            })

        # Add current host and time
        recent_hosts.append((current_host, current_time))

# -------------------------------
# Rule 2: Login outside working hours (00:00-06:00)
# -------------------------------
for idx, row in logs.iterrows():
    ts = row['timestamp']
    if time(0, 0) <= ts.time() <= time(6, 0) and row['event_id'] == 4624:  # logon event
        alerts.append({
            'timestamp': ts,
            'user': row['user'],
            'host': row['host'],
            'source_ip': row.get('source_ip', '-'),
            'dest_ip': row.get('dest_ip', '-'),
            'source_port': row.get('source_port', '-'),
            'dest_port': row.get('dest_port', '-'),
            'alert_type': 'Login outside working hours',
            'details': 'User logged in between 00:00-06:00'
        })

# -------------------------------
# Rule 3: Remote process creation
# -------------------------------
suspicious_processes = ['powershell.exe', 'cmd.exe']

for idx, row in logs.iterrows():
    if row['event_id'] == 4688 and row['process_name'].lower() in suspicious_processes:
        alerts.append({
            'timestamp': row['timestamp'],
            'user': row['user'],
            'host': row['host'],
            'source_ip': row.get('source_ip', '-'),
            'dest_ip': row.get('dest_ip', '-'),
            'source_port': row.get('source_port', '-'),
            'dest_port': row.get('dest_port', '-'),
            'alert_type': 'Remote process creation',
            'details': f"User ran {row['process_name']}: {row['command_line']}"
        })

# -------------------------------
# Rule 4: Sensitive host access by uncommon users
# -------------------------------
# Define sensitive hosts
sensitive_hosts = ['hr01', 'hr02', 'finance01', 'finance02']

# Example: allow only HR users on HR hosts and Finance users on Finance hosts
for idx, row in logs.iterrows():
    if row['host'] in sensitive_hosts:
        if (row['host'].startswith('hr') and not row['user'].lower().startswith('sara')) or \
           (row['host'].startswith('finance') and not row['user'].lower().startswith('ali')):
            alerts.append({
                'timestamp': row['timestamp'],
                'user': row['user'],
                'host': row['host'],
                'source_ip': row.get('source_ip', '-'),
                'dest_ip': row.get('dest_ip', '-'),
                'source_port': row.get('source_port', '-'),
                'dest_port': row.get('dest_port', '-'),
                'alert_type': 'Sensitive host access',
                'details': 'User accessing host they normally do not access'
            })

# -------------------------------
# Save alerts
# -------------------------------
alerts_df = pd.DataFrame(alerts)
alerts_df.to_csv(OUTPUT_FILE, index=False)
print(f"Alerts saved to {OUTPUT_FILE}")
