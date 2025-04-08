# Interactive Budget Visualization Dashboard

A powerful Streamlit-based dashboard for visualizing and analyzing budget data from Excel files, with support for local, OneDrive, and SharePoint integration.

## Features

- Upload Excel files from local storage, OneDrive, or SharePoint
- Interactive visualization of expense data by project
- Real-time budget impact analysis with toggleable budget lines
- Beautiful, responsive UI with modern design
- Export functionality for analysis results

## Setup

### Option 1: Local Setup

1. Create and activate virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: .\env\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

### Option 2: Docker Setup

1. Make sure you have Docker Desktop installed and running

2. Build and run the container:
```bash
docker-compose up --build
```

3. Access the dashboard at http://localhost:8501

## Data Format Requirements

Your Excel file should contain the following columns:
- Project
- Expense
- Budget Line
- Amount
- Date

## Microsoft Integration Setup

For OneDrive/SharePoint integration, you'll need to:
1. Register an application in Azure Portal
2. Create a `.env` file with your credentials
3. Follow the authentication flow when prompted

## Contributing

Feel free to submit issues and enhancement requests! 