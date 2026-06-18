# Start the mock LMS server
.\.venv\Scripts\Activate.ps1
uvicorn mock_lms.server:app --host 127.0.0.1 --port 8000 --reload
