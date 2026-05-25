# src/detection/ml_detection.py
import os
import pandas as pd
from sklearn.ensemble import IsolationForest

# -----------------------
# Paths
# -----------------------
FEATURE_DIR = "data/features"
FEATURE_CSV = os.path.join(FEATURE_DIR, "feature_table.csv")
ALERT_DIR = "data/alerts"
ALERT_FILE = os.path.join(ALERT_DIR, "ml_alerts.csv")
os.makedirs(ALERT_DIR, exist_ok=True)

# -----------------------
# Load Feature Table
# -----------------------
print("[+] Loading feature table...")
if os.path.exists(FEATURE_CSV):
    df = pd.read_csv(FEATURE_CSV)
    # Check if it has the new columns
    required_cols = ["num_source_ips", "num_unique_commands", "outside_work_hours"]
    if not all(col in df.columns for col in required_cols):
        import glob
        backup_files = glob.glob(os.path.join(FEATURE_DIR, "feature_table_backup_*.csv"))
        if backup_files:
            latest_backup = max(backup_files, key=os.path.getmtime)
            print(f"[+] Using backup file: {latest_backup}")
            df = pd.read_csv(latest_backup)
        else:
            print("[!] Warning: Feature table missing new columns, but no backup found")
else:
    import glob
    backup_files = glob.glob(os.path.join(FEATURE_DIR, "feature_table_backup_*.csv"))
    if backup_files:
        latest_backup = max(backup_files, key=os.path.getmtime)
        print(f"[+] Using backup file: {latest_backup}")
        df = pd.read_csv(latest_backup)
    else:
        raise FileNotFoundError(f"Feature table not found: {FEATURE_CSV}")

# -----------------------
# Select numeric columns for ML
# -----------------------
# Base columns (always present)
base_cols = ["action_count", "unique_hosts", "suspicious_process_count"]
# Enhanced columns (may be missing in old feature tables)
enhanced_cols = ["num_source_ips", "num_unique_commands", "outside_work_hours"]
# NEW: Fused Structural Features from tuned model
fused_cols = ["max_structural_anomaly_score", "num_structural_anomalies"]

# Use only columns that exist
numeric_cols = [col for col in base_cols + enhanced_cols + fused_cols if col in df.columns]
print(f"[+] Using features: {numeric_cols}")

X = df[numeric_cols].values

# -----------------------
# Dynamic ML Training (The "Old Model" logic)
# -----------------------
print("[+] Training dynamic Isolation Forest model on enhanced features...")
# Using contamination=0.003 to keep FPs extremely low, just like the old model
model = IsolationForest(contamination=0.003, random_state=42)
labels = model.fit_predict(X)
scores = model.decision_function(X)

df["anomaly_score"] = scores
df["is_anomaly"] = labels

# -----------------------
# Create alerts DataFrame
# -----------------------
anoms = df[df["is_anomaly"] == -1].copy()
alerts = []
if not anoms.empty:
    for _, r in anoms.iterrows():
        details = {col: float(r[col]) for col in numeric_cols if col in r}
        details["anomaly_score"] = float(r["anomaly_score"])
        
        alerts.append({
            "time_window": r["time_window"],
            "user": r["user"],
            "host": r.get("host_list", ""),
            "source_ip": r.get("source_ip_list", ""),
            "alert_type": "Unified ML anomaly",
            "details": str(details)
        })

if alerts:
    alerts_df = pd.DataFrame(alerts)
else:
    alerts_df = pd.DataFrame(columns=["time_window", "user", "host", "source_ip", "alert_type", "details"])

# -----------------------
# Save alerts
# -----------------------
alerts_df.to_csv(ALERT_FILE, index=False)
print(f"[OK] ML alerts saved to {ALERT_FILE}")
print(f"[OK] Number of ML alerts: {len(alerts_df)}")
