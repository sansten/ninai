import json
import os

# Check common paths for the file
paths = [
    "repos/ninai/notebooks/locomo_dataset/locomo10.json",
    "notebooks/locomo_dataset/locomo10.json",
    "locomo_dataset/locomo10.json"
]

file_path = None
for p in paths:
    if os.path.exists(p):
        file_path = p
        break

if not file_path:
    print(f"File not found in any of: {paths}")
    print(f"Current directory: {os.getcwd()}")
    # List files to help debug
    exit(1)

keywords = ["charity", "race", "mental health", "self-care", "support group", "adoption", "identity", "transgender"]

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# The data is expected to be a list of conversations or a dict with conversation keys
target_id = "locomo_001"
conv = None

if isinstance(data, list):
    for item in data:
        if item.get("conversation_id") == target_id or item.get("id") == target_id:
            conv = item
            break
elif isinstance(data, dict):
    conv = data.get(target_id) or data.get("locomo_001")

if not conv:
    print(f"Conversation {target_id} not found.")
    exit(1)

text_to_search = ""

def add_field(field_name):
    val = conv.get(field_name)
    if not val:
        return ""
    if isinstance(val, list):
        return "\n".join(str(v) for v in val) + "\n"
    return str(val) + "\n"

text_to_search += add_field("session_overview")
text_to_search += add_field("session_summary")
text_to_search += add_field("event_summary")
text_to_search += add_field("session_bullets")

if "messages" in conv:
    for msg in conv["messages"]:
        text_to_search += f"{msg.get('role', '')}: {msg.get('content', '')}\n"

lines = text_to_search.splitlines()
found = False
for line in lines:
    if any(kw.lower() in line.lower() for kw in keywords):
        print(line.strip())
        found = True

if not found:
    print("No matching lines found.")
