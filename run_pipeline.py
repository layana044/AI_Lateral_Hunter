"""
End-to-End Pipeline for Lateral Movement Detection

This script runs the complete pipeline:
1. Log Ingestion
2. Feature Engineering
3. ML Detection
4. Rule-Based Detection (if not already done)
5. Merge Alerts

Usage:
    python run_pipeline.py
"""
import os
import sys
import subprocess

def run_script(script_path, description):
    """Run a Python script and handle errors."""
    print(f"\n{'='*60}")
    print(f"[+] {description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"[OK] {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"[ERROR] {description} failed: {e}")
        return False

def main():
    """Run the complete pipeline."""
    print("\n" + "="*60)
    print("Lateral Movement Detection - End-to-End Pipeline")
    print("="*60)
    
    # Get base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)
    
    # Step 1: Log Ingestion
    if not run_script("src/ingestion/log_ingestion.py", "Log Ingestion"):
        print("\n[!] Pipeline stopped: Log ingestion failed")
        return 1
    
    # Step 2: Feature Engineering
    if not run_script("src/ml/feature_engineering.py", "Feature Engineering"):
        print("\n[!] Pipeline stopped: Feature engineering failed")
        return 1
    
    # Step 3: ML Detection
    if not run_script("src/detection/ml_detection.py", "ML Detection"):
        print("\n[!] Pipeline stopped: ML detection failed")
        return 1
    
    # Step 4: Rule-Based Detection
    if not run_script("src/detection/rule_based_detection.py", "Rule-Based Detection"):
        print("\n[!] Warning: Rule-based detection failed, but continuing...")
    
    # Step 5: Merge Alerts
    if not run_script("src/detection/merge_alerts.py", "Merge Alerts"):
        print("\n[!] Pipeline stopped: Merge alerts failed")
        return 1
    
    print("\n" + "="*60)
    print("[OK] Pipeline completed successfully!")
    print("="*60)
    print("\nOutput files:")
    print("  - Feature table: data/features/feature_table.csv")
    print("  - ML alerts: data/alerts/ml_alerts.csv")
    print("  - Rule-based alerts: data/alerts/alerts.csv")
    print("  - Merged alerts: data/alerts/all_alerts.csv")
    print("\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

