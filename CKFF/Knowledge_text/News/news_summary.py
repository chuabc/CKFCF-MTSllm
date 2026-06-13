import requests
import json
from tqdm import tqdm
import pycountry

DEEPSEEK_API_KEY = "sk-46bb769103984d8e80dfaf63f56525b7"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

def summarize_news(text, max_tokens=150):
    target_variable ="the exchange rate"      #    "weather"#"the electricity consumption"
    prompt = (
        f"Summarize the following news into a concise factual summary of around 30-50 words, "
        f"focusing only on information that may affect {target_variable}. "
        "Include relevant events, policies, economic or weather factors, and energy market reactions if applicable. "
        "Ignore unrelated details. "
        "The summary should be suitable for a knowledge base entry.\n\n"
        f"News:\n{text}"
    )


    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "deepseek-reasoner",
        "messages": [
            {"role": "system", "content": "You are a financial assistant."},#"You are an energy analyst specializing in electricity consumption. ""You are a meteorological analyst specializing in weather and climate. "
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    resp = requests.post(DEEPSEEK_API_URL, headers=headers, json=data)
    resp.raise_for_status()

    return resp.json()["choices"][0]["message"]["content"].strip()


SPECIAL_COUNTRIES = {
    "EU": "European Union",
    "TWN": "Taiwan",
    "KOR": "South Korea",
    "RUS": "Russia",
    "IRN": "Iran",
    "GBR": "United Kingdom"
}

def country_code_to_name(code):
    if not code:
        return None
    if code in SPECIAL_COUNTRIES:
        return SPECIAL_COUNTRIES[code]
    country = pycountry.countries.get(alpha_3=code)
    return country.name if country else None


def goldstein_to_intensity(goldstein):
    if goldstein is None:
        return None

    g = abs(float(goldstein))

    if g < 2:
        return "weak"
    elif g < 5:
        return "moderate"
    else:
        return "strong"


input_path = "CKFF/Knowledge_text/News/input/exchange_news_articles_diffbot.jsonl"
output_path = "CKFF/Knowledge_text/News/output/exchange_news_summary.json"

summary_entries = []

with open(input_path, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Summarizing news"):
        # line = f.readline()
        raw = json.loads(line)

        text = raw["text"]
        try:
            summary = summarize_news(text)
        except Exception as e:
            print("Error:", e)
            summary = ""

        location = []
        location_code = []
        for c in [raw.get("actor1_country"), raw.get("actor2_country")]:
            if c and c not in location_code:
                location_code.append(c)
                name = country_code_to_name(c)
                if name:
                    location.append(name)
        content = f'On {raw.get("time")}, {summary}'
        entry = {
            "domain": "weather",
            "location": "Jena_Germany", #"location": location,
            "knowledge_type": "news",
            "time": raw.get("time"),
            "title": raw.get("title"),
            "content": content,
            "news_intensity": goldstein_to_intensity(raw.get("goldstein"))
        }

        summary_entries.append(entry)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(summary_entries, f, ensure_ascii=False, indent=2)
