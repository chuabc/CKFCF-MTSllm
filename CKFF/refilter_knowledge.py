import os
import random
from datetime import timedelta
import json
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

def fetch_knowledge(now_date, day_num, df):

    now_date = pd.to_datetime(now_date)
    # 汇率和发电量
    before_window = now_date - pd.Timedelta(days=day_num)
    after_window = now_date + pd.Timedelta(days=1)
    # 天气
    # before_window = now_date - pd.Timedelta(hours=day_num)
    # after_window  = now_date + pd.Timedelta(hours=1)
    # # 风电
    # before_window = now_date - pd.Timedelta(minutes=10 *day_num)
    # after_window  = now_date + pd.Timedelta(minutes=10)
    mask = (df['start_time'] < after_window) & (df['end_time'] >= before_window)
    selected_df = df[mask]

    if selected_df.empty:
        return "No knowledge found for the prediction date."
    else:
        statements = []
        # for idx, content in enumerate(selected_df['content'].tolist()):
        #     statements.append(f"[{idx}] {content}")
        for _, row in selected_df.iterrows():
            content = row['content']
            knowledge_type = row.get('knowledge_type', '')
            statements.append(f"- ({knowledge_type}) {content}")
        return "\n".join(statements)
    

DEEPSEEK_API_KEY = "sk-46bb769103984d8e80dfaf63f56525b7"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

session = requests.Session()
session.headers.update({
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    "Content-Type": "application/json",
    "Connection": "keep-alive"
})

def gpt_refilter_knowledge(prompt):
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a knowledge filtering agent."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }
    try:
        # resp = requests.post(DEEPSEEK_API_URL, headers=headers, json=data)
        resp = session.post(DEEPSEEK_API_URL, json=data, timeout=30)
        resp.raise_for_status()

        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return str(e)
    

def is_valid_retained(item):
    rationale = item.get("rationale", "").lower()
    exclude_patterns = [
        "excluded", "not allowed", "disallowed", "not be used", 
        "rejected", "filtered out", "violates", "not permitted"
    ]
    return not any(p in rationale for p in exclude_patterns)


def refilter_knowledge_procedure(variable, KB_path, csv_file_path, new_logic):
    header = not os.path.exists(csv_file_path)

    with open(KB_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    KB_df = pd.DataFrame(data)
    KB_df['start_time'] = pd.to_datetime(KB_df['start_time'])
    KB_df['end_time'] = pd.to_datetime(KB_df['end_time'])

    # 汇率
    dates_range = pd.date_range(start="1990-01-01", end="2010-10-10")
    # 发电量
    # dates_range = pd.date_range(start="2016-07-01", end="2019-07-02")
    # 天气
    # dates_range = pd.date_range(start="2020-01-01 00:00", end="2020-12-31 23:00", freq="h")
    # 风电
    # dates_range = pd.date_range(start="2024-01-01 00:00",end="2024-04-15 23:50",freq="10T")
    for date in tqdm(dates_range):
        # 汇率和发电量
        formatted_date = date.strftime('%Y-%m-%d')
        # 天气
        # formatted_date = date.strftime('%Y-%m-%d %H:00')
        # 风电
        # formatted_date = date.strftime('%Y-%m-%d %H:%M')
        filtered_knoleg_one = fetch_knowledge(date, 3, KB_df)
        if filtered_knoleg_one == "No knowledge found for the prediction date.":
            retained_items = []
        else:
            prompt = f'''Filter relevant knowledge for predicting "{variable}" on {formatted_date}.
        
Filtering logic:
{new_logic}

Candidate knowledge entries (each line is one entry; parentheses indicate knowledge_type):
{filtered_knoleg_one}

Task:
Filter the knowledge entries according to the filtering logic for predicting {variable} on {formatted_date}.
Return ONLY the knowledge entries that should be kept.
Do NOT include any knowledge that is excluded, rejected, or disallowed by the filtering logic.

For each kept entry, output a JSON object with exactly three fields:
- content: the original knowledge text
- knowledge_type: the type shown in parentheses for this content
- rationale: a brief explanation of why this knowledge is retained and relevant for predicting the target variable (maximum 30 words)

Output requirements:
- Output strictly as a JSON array.
- Do NOT include explanations for excluded knowledge.
- If no knowledge entries satisfy the filtering logic, output an empty JSON array [].
- Do not include any additional text.
'''

            response = gpt_refilter_knowledge(prompt)
            response = response[response.find("["):response.rfind("]") + 1].replace("\n", "")
            try:
                items = json.loads(response)
                if not isinstance(items, list):
                    raise ValueError("JSON is not a list")
                retained_items = [item for item in items if is_valid_retained(item)]
            except (json.JSONDecodeError, ValueError):
                retained_items = []
        df_one = pd.DataFrame([{"pred_date": formatted_date,"knowledge": retained_items}])
        df_one.to_csv(csv_file_path, mode='a', header=header, index=False, encoding='utf-8')
        header = False