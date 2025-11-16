import os
import csv
from langchain.tools import tool
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import re
credentials_json = os.getenv("google_credentials_json")

if not credentials_json:
    raise ValueError("Environment variable 'google_credentials_json' is missing!")

credentials_dict = json.loads(credentials_json)
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(credentials_dict, scope)

def add_values(new_row,username="Interview"):
  
    try: 
        print(new_row)
        client = gspread.authorize(creds)
        sheet = client.open(username).sheet1  
        sheet.append_row(new_row)
        return "data stored"
    except Exception as e:
        return f"Failed to Add values: {str(e)}"
    
def save_qa_tool(question: str, answer: str, session_id: str | None = None, name: str | None = None, email: str | None = None, role: str | None = None  ) -> str:
    """Append a question/answer pair to interview_log.csv.

    The CSV file will be created in the same directory as this tools.py file.
    If the file is empty, a header row ["Question", "Answer"] will be written first.
    """
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, "interview_log.csv")

    os.makedirs(base_dir, exist_ok=True)

    with open(csv_path, "a+", newline='', encoding='utf-8') as f:
        f.seek(0, os.SEEK_END)
        is_empty = f.tell() == 0

        writer = csv.writer(f)
        if is_empty:
            writer.writerow(["Question", "Answer"])

        try:
            f.seek(0)
            rows = list(csv.reader(f))
            if len(rows) >= 2:
                last_row = rows[-1]
                if len(last_row) >= 2 and last_row[0] == question and last_row[1] == answer:
                    pass
        except Exception:
            pass

        writer.writerow([question, answer, session_id, name, email, role])
        from datetime import datetime

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return add_values([question, answer, session_id, name, email, role,current_time],username="Interview")
def extract_values(session_id_to_find=None):
    """
    Extracts values from the Google Sheet.
    If session_id_to_find is provided, it filters by that ID.
    Otherwise, it returns all data.
    """
    try:
        client = gspread.authorize(creds)
        sheet = client.open("Interview").sheet1
        rows = sheet.get_all_values()
        headers = [h.strip() for h in rows[0]]
        session_col_name = "Session_id" 
        if session_col_name not in headers:
            return f"Error: Column '{session_col_name}' not found. Found headers: {headers}"
        data = [dict(zip(headers, row)) for row in rows[1:]]
        if session_id_to_find is None:
            return json.dumps(data, indent=2)
        filtered_data = []
        for record in data:
            if str(record.get(session_col_name)) == str(session_id_to_find):
                filtered_data.append(record)
        
        return json.dumps(filtered_data, indent=2)

    except Exception as e:
        return f"Failed to extract values: {str(e)}"