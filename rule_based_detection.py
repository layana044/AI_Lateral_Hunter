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

# Helper function to append alerts
def add_alert(row, alert_type, details):
    alerts.append({
        'timestamp': row['timestamp'],
        'user': row['user'],
        'host': row['host'],
        'source_ip': row.get('source_ip', '-'),
        'dest_ip': row.get('dest_ip', '-'),
        'source_port': row.get('source_port', '-'),
        'dest_port': row.get('dest_port', '-'),
        'process_name': row.get('process_name', '-'),
        'alert_type': alert_type,
        'details': details
    })

# -------------------------------
# Rule 1: Multiple hosts in short time (10 min)
# -------------------------------
for user, user_logs in logs.groupby('user'):
    user_logs = user_logs.sort_values('timestamp')
    recent_hosts = []  # list of tuples (host, timestamp)

    for i in range(len(user_logs)):
        current_time = user_logs.iloc[i]['timestamp']
        current_host = user_logs.iloc[i]['host']

        # Remove hosts older than 10 minutes
        recent_hosts = [(h, t) for h, t in recent_hosts if t >= current_time - timedelta(minutes=10)]

        # Check if they accessed more than 2 distinct hosts
        unique_hosts = set(h for h, t in recent_hosts)
        unique_hosts.add(current_host)
        
        if len(unique_hosts) > 2:
            alerts.append({
                'timestamp': current_time,
                'user': user,
                'host': current_host,
                'source_ip': user_logs.iloc[i].get('source_ip', '-'),
                'dest_ip': user_logs.iloc[i].get('dest_ip', '-'),
                'source_port': user_logs.iloc[i].get('source_port', '-'),
                'dest_port': user_logs.iloc[i].get('dest_port', '-'),
                'process_name': user_logs.iloc[i].get('process_name', '-'),
                'alert_type': 'Multiple hosts in short time',
                'details': f"User accessed {len(unique_hosts)} distinct hosts ({unique_hosts}) within 10 minutes"
            })
            recent_hosts = [] # Clear the tracker to prevent alert storms
        else:
            # Add current host and time only if we didn't just clear it
            recent_hosts.append((current_host, current_time))

# -------------------------------
# Rule 2: Login outside working hours (00:00-06:00)
# -------------------------------
for idx, row in logs.iterrows():
    ts = row['timestamp']
    if time(0, 0) <= ts.time() <= time(6, 0) and row['event_id'] == 4624:  # logon event
        add_alert(row, 'Login outside working hours', 'User logged in between 00:00-06:00')

# -------------------------------
# Rule 3: Remote process creation
# -------------------------------
suspicious_processes = ['powershell.exe', 'cmd.exe', 'psexec.exe', 'psexesvc.exe', 'wmic.exe']

for idx, row in logs.iterrows():
    if row['event_id'] == 4688:
        proc_name = str(row['process_name']).lower()
        if any(sp in proc_name for sp in suspicious_processes):
            add_alert(row, 'Remote process creation', f"User ran {row['process_name']}: {row['command_line']}")

# -------------------------------
# Rule 4: Generalized High-Value Target (HVT) Access
# -------------------------------
for idx, row in logs.iterrows():
    host = str(row['host']).upper()
    user = str(row['user']).lower()
    
    # Check if host is an HVT (Domain Controller, Database, etc)
    if 'DC' in host or 'SQL' in host or 'DB' in host or 'ADMIN' in host:
        # Flag if user is not an admin or service account
        if 'admin' not in user and 'svc' not in user and user != 'sysmon_user':
            add_alert(row, 'Sensitive host access', f"Standard user {user} accessed High-Value Target {host}")

# -------------------------------
# Rule 5: Credential Dumping (Event ID 10) - T1003
# -------------------------------
for idx, row in logs.iterrows():
    if row['event_id'] == 10:  # ProcessAccess
        target_image = str(row.get('target_image', '')).lower()
        if 'lsass.exe' in target_image or 'lsass.exe' in str(row['command_line']).lower():
            add_alert(row, 'Credential Dumping (LSASS)', f"Suspicious access to lsass.exe by {row['process_name']}")

# -------------------------------
# Rule 6: Named Pipe Lateral Tool Transfer (Event IDs 17/18) - T1570
# -------------------------------
suspicious_pipes = ['psexesvc', 'msagent', 'postex', 'status', 'msrpc']

for idx, row in logs.iterrows():
    if row['event_id'] in [17, 18]:  # Pipe Created or Connected
        pipe_name = str(row.get('pipe_name', row.get('command_line', ''))).lower()
        if any(pipe in pipe_name for pipe in suspicious_pipes):
            add_alert(row, 'Malicious Named Pipe', f"Suspicious named pipe activity: {pipe_name}")

# -------------------------------
# Rule 7: Suspicious Network Connections (Event ID 3) - T1021
# -------------------------------
sensitive_ports = ['445', '3389', '5985', '5986'] # SMB, RDP, WinRM
standard_browsers = ['chrome.exe', 'firefox.exe', 'msedge.exe', 'iexplore.exe']

for idx, row in logs.iterrows():
    if row['event_id'] == 3:  # Network Connection
        dest_port = str(row.get('dest_port', '-'))
        proc_name = str(row['process_name']).lower()
        
        # Flag if a non-browser process connects to a sensitive port
        if dest_port in sensitive_ports and not any(b in proc_name for b in standard_browsers):
            # Ignore System process connecting to SMB, which is normal
            if not (proc_name == 'system' and dest_port == '445'):
                add_alert(row, 'Suspicious Network Connection', f"Process {row['process_name']} connected to port {dest_port}")

# -------------------------------
# Rule 8: Process Injection (Event ID 8) - T1055
# -------------------------------
for idx, row in logs.iterrows():
    if row['event_id'] == 8:  # CreateRemoteThread
        add_alert(row, 'Process Injection', f"CreateRemoteThread detected by {row['process_name']}")

# -------------------------------
# Save alerts
# -------------------------------
alerts_df = pd.DataFrame(alerts)

# Drop duplicates if any (e.g. if an event triggered multiple overlapping rules)
if not alerts_df.empty:
    alerts_df = alerts_df.drop_duplicates(subset=['timestamp', 'user', 'alert_type'])

alerts_df.to_csv(OUTPUT_FILE, index=False)
print(f"Alerts saved to {OUTPUT_FILE}")
