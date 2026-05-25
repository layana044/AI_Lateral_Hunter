import csv
import random
from datetime import datetime, timedelta
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
os.makedirs(RAW_DIR, exist_ok=True)

# -----------------------------
# Configuration
# -----------------------------
NUM_NORMAL = 850
NUM_ABNORMAL = 15

# Fields required by log_ingestion.py
HEADERS = [
    'UtcTime', 'User', 'Computer', 'EventID', 'Image', 
    'CommandLine', 'SourceIp', 'DestinationIp', 'SourcePort', 'DestinationPortName'
]

# Assets
USERS = ['jdoe', 'asmith', 'bwayne', 'ckent']
ADMINS = ['admin_jdoe', 'svc_deploy']
USER_WORKSTATIONS = {
    'jdoe': 'WKSTN-101',
    'asmith': 'WKSTN-102',
    'bwayne': 'WKSTN-103',
    'ckent': 'WKSTN-104'
}
SERVERS = ['SRV-FS01', 'SRV-WEB01']
HVTS = ['SRV-DC01', 'SRV-SQL01', 'SRV-DB01'] # High Value Targets

NORMAL_PROCESSES = [
    r'C:\Windows\System32\svchost.exe',
    r'C:\Windows\explorer.exe',
    r'C:\Program Files\Google\Chrome\Application\chrome.exe',
    r'C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE'
]

def random_time(start, end):
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = random.randrange(int_delta)
    return start + timedelta(seconds=random_second)

def generate_normal_log(start_time, end_time):
    ts = random_time(start_time, end_time)
    user = random.choice(USERS)
    comp = USER_WORKSTATIONS[user]  # Normal users stay on their own workstation!
    
    event_id = random.choice([1, 3, 4624])
    
    image = '-'
    cmd = '-'
    sip = '-'
    dip = '-'
    sport = '-'
    dport = '-'
    
    if event_id == 4624: # Logon
        pass # just keep dashes
    elif event_id == 1: # Process Create
        image = random.choice(NORMAL_PROCESSES)
    elif event_id == 3: # Network
        image = r'C:\Program Files\Google\Chrome\Application\chrome.exe'
        sip = f"192.168.1.{random.randint(10, 50)}"
        dip = f"{random.randint(10, 200)}.{random.randint(1, 255)}.1.1"
        sport = str(random.randint(10000, 60000))
        dport = '443'
        
    return [
        ts.strftime('%Y-%m-%d %H:%M:%S'), user, comp, event_id, image, 
        cmd, sip, dip, sport, dport
    ]

def generate_abnormal_logs(start_time):
    logs = []
    attacker = 'jdoe' # compromised user
    
    # 1. HVT Access by non-admin (Rule 4)
    ts = start_time + timedelta(minutes=5)
    logs.append([
        ts.strftime('%Y-%m-%d %H:%M:%S'), attacker, random.choice(HVTS), 4624, '-', 
        '-', '192.168.1.15', '-', '-', '-'
    ])
    
    # 2. Credential Dumping (Event 10, Rule 5)
    ts += timedelta(minutes=2)
    logs.append([
        ts.strftime('%Y-%m-%d %H:%M:%S'), attacker, 'WKSTN-102', 10, r'C:\Users\jdoe\Downloads\mimikatz.exe', 
        'mimikatz.exe privilege::debug sekurlsa::logonpasswords', '-', '-', '-', '-'
    ])
    # To match the rule exactly, we put lsass.exe in command line or image.
    ts += timedelta(seconds=15)
    logs.append([
        ts.strftime('%Y-%m-%d %H:%M:%S'), attacker, 'WKSTN-102', 10, r'C:\Windows\System32\taskmgr.exe', 
        'taskmgr.exe dump lsass.exe', '-', '-', '-', '-'
    ])

    # 3. Named Pipe Lateral Tool Transfer (Event 17/18, Rule 6)
    ts += timedelta(minutes=5)
    logs.append([
        ts.strftime('%Y-%m-%d %H:%M:%S'), attacker, 'WKSTN-102', 17, r'C:\Windows\System32\services.exe', 
        r'\psexesvc', '-', '-', '-', '-'
    ])
    
    # 4. Suspicious Network Connection (Event 3, Rule 7)
    # cmd.exe connecting to port 445 (SMB)
    ts += timedelta(minutes=2)
    logs.append([
        ts.strftime('%Y-%m-%d %H:%M:%S'), attacker, 'WKSTN-102', 3, r'C:\Windows\System32\cmd.exe', 
        '-', '192.168.1.15', '192.168.1.100', '49552', '445'
    ])
    
    # 5. Process Injection (Event 8, Rule 8)
    ts += timedelta(minutes=1)
    logs.append([
        ts.strftime('%Y-%m-%d %H:%M:%S'), attacker, 'WKSTN-102', 8, r'C:\Users\jdoe\payload.exe', 
        '-', '-', '-', '-', '-'
    ])

    # 6. Remote process creation (Rule 3)
    ts += timedelta(minutes=1)
    logs.append([
        ts.strftime('%Y-%m-%d %H:%M:%S'), attacker, 'WKSTN-102', 4688, r'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe', 
        'powershell.exe -nop -w hidden -EncodedCommand JABzAD0ATg', '-', '-', '-', '-'
    ])
    
    # 7. Multiple hosts in short time (Rule 1)
    ts += timedelta(minutes=1)
    for h in ['SRV-WEB01', 'SRV-FS01', 'WKSTN-104']:
        logs.append([
            ts.strftime('%Y-%m-%d %H:%M:%S'), attacker, h, 4624, '-', 
            '-', '192.168.1.15', '-', '-', '-'
        ])
        ts += timedelta(seconds=30)
        
    # 8. Login outside working hours (Rule 2)
    # Force a time at 3 AM
    night_ts = start_time.replace(hour=3, minute=15)
    logs.append([
        night_ts.strftime('%Y-%m-%d %H:%M:%S'), attacker, 'WKSTN-102', 4624, '-', 
        '-', '192.168.1.15', '-', '-', '-'
    ])

    # Pad with some extra abnormal to reach NUM_ABNORMAL
    while len(logs) < NUM_ABNORMAL:
        ts += timedelta(minutes=1)
        logs.append([
            ts.strftime('%Y-%m-%d %H:%M:%S'), attacker, 'WKSTN-102', 3, r'C:\Windows\System32\powershell.exe', 
            '-', '192.168.1.15', '10.10.10.10', '55555', '3389'
        ])
        
    return logs

def main():
    filename = f"sysmon_{NUM_NORMAL}normal_{NUM_ABNORMAL}abnormal.csv"
    filepath = os.path.join(BASE_DIR, filename)
    
    start_time = datetime(2025, 11, 20, 9, 0, 0)
    end_time = datetime(2025, 11, 20, 17, 0, 0)
    
    logs = []
    
    # Generate normal
    for _ in range(NUM_NORMAL):
        logs.append(generate_normal_log(start_time, end_time))
        
    # Generate abnormal
    abnormal_logs = generate_abnormal_logs(start_time)
    logs.extend(abnormal_logs)
    
    # Sort by time
    logs.sort(key=lambda x: x[0])
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS)
        writer.writerows(logs)
        
    print(f"Successfully generated {filepath}")
    print(f"Total rows: {len(logs)}")
    print(f"Normal: {NUM_NORMAL}, Abnormal: {NUM_ABNORMAL}")

if __name__ == '__main__':
    main()
