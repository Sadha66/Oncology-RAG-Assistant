import json

with open("data/cleaned_output.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(type(data))
print(data[0])