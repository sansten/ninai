import requests
import json
import os

BASE_URL = "https://admin.ninai.sansten.com/api/v1"
EMAIL = "demo@ninai.dev"
PASSWORD = "demo1234"
RUN_TAG = "locomo-full-20260517-001804-7b59f189"
CONV_ID = "locomo_001"

def main():
    session = requests.Session()
    login_resp = session.post(f"{BASE_URL}/auth/login", json={"email": EMAIL, "password": PASSWORD})
    login_resp.raise_for_status()
    token = login_resp.json()["access_token"]
    session.headers.update({"Authorization": f"Bearer {token}"})

    print(f"Logged in as {EMAIL}")
    
    # Check if we can find ANY memories first
    resp = session.get(f"{BASE_URL}/memories", params={"limit": 5})
    print(f"Sample memory tags: {[m.get('tags') for m in resp.json().get('items', [])]}")

    c_run = session.get(f"{BASE_URL}/memories", params={"tags": [RUN_TAG], "limit": 1}).json().get("total", 0)
    c_conv = session.get(f"{BASE_URL}/memories", params={"tags": [CONV_ID], "limit": 1}).json().get("total", 0)
    c_both = session.get(f"{BASE_URL}/memories", params={"tags": [RUN_TAG, CONV_ID], "limit": 1}).json().get("total", 0)
    print(f"Counts - Run ({RUN_TAG}): {c_run}, Conv ({CONV_ID}): {c_conv}, Both: {c_both}")

    questions = [
        "What did the charity race raise awareness for?",
        "When did Caroline go to the LGBTQ support group?",
        "What did Caroline research?"
    ]

    for q in questions:
        print(f"\n--- Question: {q} ---")
        # Try search with ONLY RUN_TAG to see if that works
        for use_graph in [True, False]:
            params = [
                ('query', q),
                # No tags at all to see if we get anything
                ('limit', 5),
                ('threshold', 0.0),
                ('use_graph', str(use_graph).lower()),
                ('hybrid', 'true')
            ]
            res = session.get(f"{BASE_URL}/memories/search", params=params)
            hits = res.json().get("items", []) if res.status_code == 200 else []
            print(f"NoTags Graph={use_graph} Hits: {len(hits)}")
            for h in hits[:1]:
                print(f"  Score: {h.get('score')} | Tags: {h.get('tags')} | {h.get('content')[:80]}...")

if __name__ == "__main__":
    main()
