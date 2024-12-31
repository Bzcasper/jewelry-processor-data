import asyncio
import json
import logging
import os
import queue
import shutil
import tempfile
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp
import magic
from flask import Flask, current_app, jsonify, render_template, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from jewelry_detector import CacheManager, CloudStorageManager, EnhancedJewelryProcessor
from tenacity import retry, stop_after_attempt, wait_exponential
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add to app configuration
app.config.update(
    STATIC_FOLDER=Path('src/web/static/dist')
)


@dataclass
class ProcessingJob:
    """Container for processing job information."""
    job_id: str
    files: List[Path]
    settings: Dict
    status: str = "pending"
    progress: int = 0
    total_files: int = 0
    results: List[Dict] = None
    error: Optional[str] = None


class ProcessingManager:
    """Manager for processing jobs and queues."""

    def __init__(self):
        self.queue = queue.Queue()
        self.jobs = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.socketio = None

    def set_socketio(self, socketio):
        """Set SocketIO instance for real-time updates."""
        self.socketio = socketio

    def add_job(self, job: ProcessingJob) -> str:
        """Add new processing job to queue."""
        self.jobs[job.job_id] = job
        self.queue.put(job)
        return job.job_id

    def get_job_status(self, job_id: str) -> Dict:
        """Get current status of a job."""
        job = self.jobs.get(job_id)
        if not job:
            return {"status": "not_found"}

        return {
            "status": job.status,
            "progress": job.progress,
            "total_files": job.total_files,
            "results": job.results,
            "error": job.error
        }

    async def process_job(self, job: ProcessingJob):
        """Process a single job."""
        try:
            processor = EnhancedJewelryProcessor(
                input_dir="uploads",
                output_dir="processed",
                cloud_storage=CloudStorageManager(
    current_app.config['CLOUD_STORAGE_BUCKET']),
                cache_manager=CacheManager(current_app.config['REDIS_URL']),
                settings=job.settings
            )

            job.status = "processing"
            job.total_files = len(job.files)
            self._emit_progress(job)

            results = []
            for idx, file_path in enumerate(job.files, 1):
                try:
                    result = await processor.process_image(file_path)
                    results.append(result)

                    job.progress = idx
                    self._emit_progress(job)

                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    self._emit_error(job, file_path, str(e))

            job.status = "completed"
            job.results = results
            self._emit_progress(job)

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            self._emit_error(job, None, str(e))

    def _emit_progress(self, job: ProcessingJob):
        """Emit progress update via WebSocket."""
        if self.socketio:
            self.socketio.emit('progress', {
                'job_id': job.job_id,
                'status': job.status,
                'progress': job.progress,
                'total': job.total_files
            })

    def _emit_error(self, job: ProcessingJob,
                    file_path: Optional[Path], error: str):
        """Emit error via WebSocket."""
        if self.socketio:
            self.socketio.emit('error', {
                'job_id': job.job_id,
                'file': str(file_path) if file_path else None,
                'error': error
            })


# Initialize Flask application
app = Flask(__name__)
    static_folder = 'static/dist',
    static_url_path = '/static'
)
app.wsgi_app= ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
CORS(app)
app = Flask(__name__,


# Configure Flask
app.config.update(
    MAX_CONTENT_LENGTH=100 * 1024 * 1024,  # 100MB max file size
    UPLOAD_FOLDER=Path('uploads'),
    PROCESSED_FOLDER=Path('processed'),
    ALLOWED_EXTENSIONS={'png', 'jpg', 'jpeg', 'gif', 'zip'},
    CLOUD_STORAGE_BUCKET=os.getenv('CLOUD_STORAGE_BUCKET'),
    REDIS_URL=os.getenv('REDIS_URL', 'redis://localhost:6379')
)

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize ProcessingManager
processing_manager = ProcessingManager()
processing_manager.set_socketio(socketio)

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower(
           ) in app.config['ALLOWED_EXTENSIONS']

@retry(stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=1, min=4, max=10))
async def validate_file(file_path: Path) -> Tuple[bool, str]:
    """Validate file content and type."""
    try:
        mime = magic.Magic(mime=True)
        file_type = mime.from_file(str(file_path))

        if file_type.startswith('image/'):
            return True, file_type
        elif file_type == 'application/zip':
            return True, file_type

        return False, f"Invalid file type: {file_type}"

    except Exception as e:
        logger.error(f"File validation error: {str(e)}")
        return False, str(e)

async def handle_upload(file) -> Tuple[bool, str]:
    """Handle file upload with validation."""
    if not allowed_file(file.filename):
        return False, "File type not allowed"

    try:
        filename = secure_filename(file.filename)
        temp_dir = Path(tempfile.mkdtemp())
        file_path = temp_dir / filename
        file.save(str(file_path))

        is_valid, file_type = await validate_file(file_path)
        if not is_valid:
            shutil.rmtree(temp_dir)
            return False, file_type

        return True, str(temp_dir)

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return False, str(e)

@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')

@app.route('/api/process', methods=['POST'])
async def process_request():
    """Handle file upload and processing request."""
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    try:
        files = request.files.getlist('files')
        settings = json.loads(request.form.get('settings', '{}'))

        processed_files = []
        for file in files:
            if file.filename:
                success, result = await handle_upload(file)
                if not success:
                    return jsonify({'error': result}), 400

                temp_dir = Path(result)
                if file.filename.endswith('.zip'):
                    with zipfile.ZipFile(temp_dir / file.filename, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    processed_files.extend(
                        [f for f in temp_dir.glob('**/*') if f.is_file()])
                else:
                    processed_files.append(temp_dir / file.filename)

        # Create and queue job
        job = ProcessingJob(
            job_id=f"job_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            files=processed_files,
            settings=settings
        )

        job_id = processing_manager.add_job(job)

        # Start processing in thread
        threading.Thread(
            target=asyncio.run,
            args=(processing_manager.process_job(job),),
            daemon=True
        ).start()

        return jsonify({
            'message': 'Processing started',
            'job_id': job_id
        })

    except Exception as e:
        logger.error(f"Processing request error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status/<job_id>')
def get_job_status(job_id: str):
    """Get job status."""
    status = processing_manager.get_job_status(job_id)
    return jsonify(status)

@app.errorhandler(Exception)
def handle_error(error):
    """Global error handler."""
    logger.error(f"Unhandled error: {str(error)}")
    return jsonify({'error': str(error)}), 500

if __name__ == '__main__':
    # Create required directories
    app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)
    app.config['PROCESSED_FOLDER'].mkdir(exist_ok=True)

    # Run application
    socketio.run(app, host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
