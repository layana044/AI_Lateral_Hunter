"""
Visualization script for lateral movement detection alerts.

Creates visualizations showing:
- Alerts per user
- Alerts per host
- Alerts over time
- Alert type distribution
- ML vs Rule-based comparison
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Set style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ALERTS_FILE = os.path.join(BASE_DIR, "data", "alerts", "all_alerts.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "alerts", "visualizations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("[+] Loading alerts...")
df = pd.read_csv(ALERTS_FILE)

# Convert timestamps
df['time_window'] = pd.to_datetime(df['time_window'])
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"[+] Loaded {len(df)} alerts")

# -----------------------
# 1. Alerts per User
# -----------------------
print("[+] Creating alerts per user visualization...")
fig, ax = plt.subplots(figsize=(10, 6))
user_counts = df['user'].value_counts()
bars = ax.bar(user_counts.index, user_counts.values, color=['#FF6B6B', '#4ECDC4', '#95E1D3', '#F38181'])
ax.set_xlabel('User', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Alerts', fontsize=12, fontweight='bold')
ax.set_title('Alerts per User', fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'alerts_per_user.png'), dpi=300, bbox_inches='tight')
print(f"[OK] Saved: alerts_per_user.png")
plt.close()

# -----------------------
# 2. Alerts per Host
# -----------------------
print("[+] Creating alerts per host visualization...")
# Filter out empty hosts
host_df = df[df['host'].notna() & (df['host'] != '')]
if not host_df.empty:
    fig, ax = plt.subplots(figsize=(10, 6))
    host_counts = host_df['host'].value_counts()
    bars = ax.bar(host_counts.index, host_counts.values, color='#95E1D3')
    ax.set_xlabel('Host', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Alerts', fontsize=12, fontweight='bold')
    ax.set_title('Alerts per Host', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'alerts_per_host.png'), dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: alerts_per_host.png")
    plt.close()
else:
    print("[!] No host data available for visualization")

# -----------------------
# 3. Alerts Over Time
# -----------------------
print("[+] Creating alerts over time visualization...")
fig, ax = plt.subplots(figsize=(14, 6))
df_sorted = df.sort_values('time_window')
time_counts = df_sorted.groupby(df_sorted['time_window'].dt.floor('10min')).size()

ax.plot(time_counts.index, time_counts.values, marker='o', linewidth=2, markersize=8, color='#FF6B6B')
ax.fill_between(time_counts.index, time_counts.values, alpha=0.3, color='#FF6B6B')
ax.set_xlabel('Time Window', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Alerts', fontsize=12, fontweight='bold')
ax.set_title('Alerts Over Time', fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'alerts_over_time.png'), dpi=300, bbox_inches='tight')
print(f"[OK] Saved: alerts_over_time.png")
plt.close()

# -----------------------
# 4. Alert Type Distribution
# -----------------------
print("[+] Creating alert type distribution...")
fig, ax = plt.subplots(figsize=(10, 6))
alert_type_counts = df['alert_type'].value_counts()
colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#F38181', '#AA96DA']
bars = ax.bar(alert_type_counts.index, alert_type_counts.values, color=colors[:len(alert_type_counts)])
ax.set_xlabel('Alert Type', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Alerts', fontsize=12, fontweight='bold')
ax.set_title('Alert Type Distribution', fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'alert_type_distribution.png'), dpi=300, bbox_inches='tight')
print(f"[OK] Saved: alert_type_distribution.png")
plt.close()

# -----------------------
# 5. ML vs Rule-based Comparison
# -----------------------
print("[+] Creating ML vs Rule-based comparison...")
fig, ax = plt.subplots(figsize=(8, 6))
source_counts = df['source'].value_counts()
colors = ['#4ECDC4', '#FF6B6B']
bars = ax.bar(source_counts.index, source_counts.values, color=colors)
ax.set_xlabel('Detection Method', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Alerts', fontsize=12, fontweight='bold')
ax.set_title('ML vs Rule-Based Detection Comparison', fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ml_vs_rule_based.png'), dpi=300, bbox_inches='tight')
print(f"[OK] Saved: ml_vs_rule_based.png")
plt.close()

# -----------------------
# 6. User Activity Timeline
# -----------------------
print("[+] Creating user activity timeline...")
fig, ax = plt.subplots(figsize=(14, 6))
users = df['user'].unique()
colors_map = {'ali': '#FF6B6B', 'sara': '#4ECDC4'}

for user in users:
    user_df = df[df['user'] == user].sort_values('time_window')
    user_counts = user_df.groupby(user_df['time_window'].dt.floor('10min')).size()
    color = colors_map.get(user.lower(), '#95E1D3')
    ax.plot(user_counts.index, user_counts.values, marker='o', linewidth=2, 
            markersize=8, label=user, color=color)

ax.set_xlabel('Time Window', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Alerts', fontsize=12, fontweight='bold')
ax.set_title('User Activity Timeline', fontsize=14, fontweight='bold', pad=20)
ax.legend(title='User', fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'user_activity_timeline.png'), dpi=300, bbox_inches='tight')
print(f"[OK] Saved: user_activity_timeline.png")
plt.close()

# -----------------------
# Summary Statistics
# -----------------------
print("\n" + "="*60)
print("ALERT SUMMARY STATISTICS")
print("="*60)
print(f"\nTotal Alerts: {len(df)}")
print(f"\nBy User:")
print(df['user'].value_counts().to_string())
print(f"\nBy Alert Type:")
print(df['alert_type'].value_counts().to_string())
print(f"\nBy Detection Method:")
print(df['source'].value_counts().to_string())
print(f"\nTime Range: {df['time_window'].min()} to {df['time_window'].max()}")
print("\n" + "="*60)

print(f"\n[OK] All visualizations saved to: {OUTPUT_DIR}")
print("\nGenerated files:")
for file in os.listdir(OUTPUT_DIR):
    if file.endswith('.png'):
        print(f"  - {file}")

