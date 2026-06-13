import os
import random
from datetime import timedelta
import json
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEEPSEEK_API_KEY = ""
DEEPSEEK_API_URL = ""

session = requests.Session()
session.headers.update({
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    "Content-Type": "application/json",
    "Connection": "keep-alive"
})

def discretize_contribution(delta: float) -> str:
    if delta <= -0.1:
        return "strongly_negative"
    elif delta < -0.01:
        return "weak_negative"
    elif delta <= 0.01:
        return "neutral"
    else:
        return "positive"


def logic_evaluation(prompt):
    # headers = {
    #     "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    #     "Content-Type": "application/json"
    # }

    data = {
        "model": "deepseek-reasoner",
        "messages": [
            {"role": "system", "content": "You are a rule-refinement agent that outputs only newly added or tightened filtering constraints."},
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
        return None


def logic_batch_summarize_generate(prompt):
    # headers = {
    #     "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    #     "Content-Type": "application/json"
    # }

    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a rule-integration agent that summarizes newly identified constraints."},
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
        return None
    

def new_logic_upgrade(prompt):
    # headers = {
    #     "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    #     "Content-Type": "application/json"
    # }

    data = {
        "model": "deepseek-reasoner",
        "messages": [
            {"role": "system", "content": "You are an agent that generates updated knowledge filtering logic based on the original logic and new constraints."},
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
        return None
    