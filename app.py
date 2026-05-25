import os
import glob
import subprocess
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='dashboard', static_url_path='/')
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')

# Ensure directories exist
os.makedirs(DATA_RAW_DIR, exist_ok=True)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/data/<path:path>')
def serve_data(path):
    # Important: allow serving from the data directory for CSVs and images
    return send_from_directory(os.path.join(BASE_DIR, 'data'), path)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and file.filename.endswith('.csv'):
            # 1. Clear existing files in data/raw
            existing_files = glob.glob(os.path.join(DATA_RAW_DIR, '*.csv'))
            for f in existing_files:
                try:
                    os.remove(f)
                except Exception as e:
                    print(f"Error removing {f}: {e}")
            
            # Clear old outputs to ensure clean analysis
            for folder in ['features', 'alerts']:
                folder_path = os.path.join(BASE_DIR, 'data', folder)
                if os.path.exists(folder_path):
                    for f in glob.glob(os.path.join(folder_path, '*.csv')):
                        try:
                            os.remove(f)
                        except:
                            pass
                    
            # 2. Save new file
            filename = secure_filename(file.filename)
            file_path = os.path.join(DATA_RAW_DIR, filename)
            file.save(file_path)
            
            # 3. Run Pipeline
            try:
                subprocess.run(['python', 'run_pipeline.py'], cwd=BASE_DIR, check=True)
                subprocess.run(['python', 'src/utils/visualize_alerts.py'], cwd=BASE_DIR, check=True)
                return jsonify({'status': 'success', 'message': 'Analysis complete'})
            except subprocess.CalledProcessError as e:
                return jsonify({'error': f'Pipeline failed: {str(e)}'}), 500
                
        return jsonify({'error': 'Invalid file format. Please upload a CSV file.'}), 400
    except Exception as e:
        import traceback
        return jsonify({'error': 'Exception: ' + str(e) + ' Traceback: ' + traceback.format_exc()}), 500

if __name__ == '__main__':
    # Add a cache-busting header to ensure images refresh
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.run(host='0.0.0.0', port=8085, debug=True, use_reloader=False)
