# Arabic SQL Query Assistant

A web-based system that converts Arabic questions into SQL queries using AI technology (SQLCoder-7B).

## Features
- Convert Arabic questions to SQL queries
- Web interface for easy interaction
- MySQL database support
- REST API available

## Setup & Run

### Requirements
- Python 3.8+
- MySQL Database
- [Google Colab](https://colab.research.google.com/) for model execution (Due to high memory requirements)

### Installation Steps

1. First, create a Python virtual environment (recommended):
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

2. Install all required packages:
```bash
pip install -r requirements.txt
```

3. Set up your MySQL database and update the configuration in `api.py`:
```python
mysql_config = {
    "host": "your_database_host",
    "user": "your_username",
    "password": "your_password",
    "database": "your_database_name",
    "port": 3306
}
```

4. Important Note:
   - The AI model (SQLCoder-7B) requires significant RAM (>16GB)
   - For optimal performance, use Google Colab or a machine with GPU
   - The model will run on CPU but will be slower

5. Run the application:
```bash
python api.py
```

6. Open the URL shown in terminal to use the web interface

## Project Files
- `ai_query.py` - Main AI logic and database operations
- `api.py` - FastAPI web server and API endpoints
- `templates/index.html` - Web user interface
- `requirements.txt` - All required Python packages

## Troubleshooting
- If you get memory errors, try running the model on Google Colab
- For database connection issues, check your MySQL credentials
- Make sure all packages are installed correctly
- For GPU users, ensure CUDA is properly installed

## License
MIT License