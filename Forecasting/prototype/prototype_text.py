import os
# from openai import OpenAI
import numpy as np
import pandas as pd
from tqdm import tqdm
import requests
import time


def build_timeseries_prompt(ts, proto_id):
    """
    ts: numpy array, shape [L]
    """
    mean = ts.mean()
    std = ts.std()
    min_v = ts.min()
    max_v = ts.max()

    # 简单趋势（线性斜率）
    x = np.arange(len(ts))
    slope = np.polyfit(x, ts, 1)[0]

    trend_desc = (
        "increasing" if slope > 0 else
        "decreasing" if slope < 0 else
        "stable"
    )
    ts_str = ", ".join(f"{v:.3f}" for v in ts)
    prompt = f"""
        You are an expert in time series analysis.

        The following is a time series: [{ts_str}].
        
        Statistical properties:
        - Mean value: {mean:.3f}
        - Standard deviation: {std:.3f}
        - Minimum value: {min_v:.3f}
        - Maximum value: {max_v:.3f}
        - Overall trend: {trend_desc}
        
        Task:
        Write exactly 1–2 concise sentences describing the temporal behavior of this time series, 
        focusing on its trend, periodicity, fluctuation, and possible pattern. 
        Do NOT mention raw numerical values explicitly.
        Do NOT output analysis, reasoning, or step-by-step process.
        """
    return prompt.strip()

'''     
        Example:
        "The time series exhibits an overall increasing trend with small fluctuation."
        "The time series exhibits cyclic behavior with with a medium-length period."'''
DEEPSEEK_API_KEY = ""
DEEPSEEK_API_URL = ""

def call_deepseek(prompt):
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "deepseek-reasoner",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    resp = requests.post(DEEPSEEK_API_URL, headers=headers, json=data)
    resp.raise_for_status()

    return resp.json()["choices"][0]["message"]["content"].strip()


prototypes = np.load("prototype/exchange/100/prototype_centers.npy")
print(prototypes.shape)  # (1000, L)
records = []

for i in tqdm(range(len(prototypes))):
    ts = prototypes[i]
    prompt = build_timeseries_prompt(ts, i)

    try:
        description = call_deepseek(prompt)
    except Exception as e:
        description = f"ERROR: {e}"

    records.append({
        "prototype_id": i,
        "time_series": ts.tolist(),   # 或保存为字符串
        "text_description": description
    })
    time.sleep(0.5)

df = pd.DataFrame(records)
df.to_csv("prototype/exchange/100/reasoner_timeseries_prototype_with_text.csv", index=False)