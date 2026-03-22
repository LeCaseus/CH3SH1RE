import json
from backend.memory import upsert_fact

with open("seed_facts.json", "r") as f:
    data = json.load(f)

for key, value in data["facts"].items():
    upsert_fact(key, value)

for key, value in data["instructions"].items():
    upsert_fact(key, value)

print("Facts seeded.")
