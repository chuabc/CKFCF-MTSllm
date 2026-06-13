import json

file1 = "CKFF/Knowledge_text/Knowledge/weather_knowledge.json"
file2 = "CKFF/Knowledge_text/News/output/weather_news_summary.json"

output_file = "CKFF/Knowledge_text/Knowledge_base/weather_knowledge_base.json"

# file1 = "CKFF/Knowledge_text/Knowledge_base/exchange_knowledge_base.json"
# file2 = "CKFF/Knowledge_text/Knowledge_base/ecl_knowledge_base.json"
# file3 = "CKFF/Knowledge_text/Knowledge_base/weather_knowledge_base.json"
# file4 = "CKFF/Knowledge_text/Knowledge_base/wind_knowledge_base.json"

# output_file = "CKFF/Knowledge_text/Knowledge_base/ALL_knowledge_base.json"

# file1 = "CKFF/Knowledge_text/Standard_Knowledge_base/exchange_knowledge_base.json"
# file2 = "CKFF/Knowledge_text/Standard_Knowledge_base/ecl_knowledge_base.json"
# file3 = "CKFF/Knowledge_text/Standard_Knowledge_base/weather_knowledge_base.json"
# file4 = "CKFF/Knowledge_text/Standard_Knowledge_base/wind_knowledge_base.json"

# output_file = "CKFF/Knowledge_text/Standard_Knowledge_base/ALL_knowledge_base.json"

with open(file1, "r", encoding="utf-8") as f:
    data1 = json.load(f)

with open(file2, "r", encoding="utf-8") as f:
    data2 = json.load(f)

# with open(file3, "r", encoding="utf-8") as f:
#     data3 = json.load(f)

# with open(file4, "r", encoding="utf-8") as f:
#     data4 = json.load(f)
# # 合并
# merged_data = data1 + data2 + data3 + data4
merged_data = data1 + data2

# 保存
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=2)

print(f"合并完成，共 {len(merged_data)} 条，已保存到 {output_file}")
