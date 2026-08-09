"""
Delete ALL memories tagged 'locomo' from the org.
Run this before a fresh locomo ingest to clear the content-dedup cache.
"""
import sys
import time
import urllib.request
import json

from ninai import NinaiClient

BASE_URL = 'https://admin.ninai.sansten.com/api/v1'
EMAIL    = 'demo@ninai.dev'
PASSWORD = 'demo1234'
ORG_SLUG = 'default'

client = NinaiClient(base_url=BASE_URL, timeout=300.0)
client.login(email=EMAIL, password=PASSWORD, org_slug=ORG_SLUG)
token = client._access_token or ''
print(f'Authenticated. Token len={len(token)}')


def _list_page(tag, page, page_size=100):
    try:
        r = client.memories.list(tags=[tag], page=page, page_size=page_size)
        return r.items, r.has_more
    except Exception as e:
        print(f'  WARN list page {page}: {e}')
        return [], False


def _batch_delete(ids, batch_size=500):
    deleted = failed = 0
    for i in range(0, len(ids), batch_size):
        batch = ids[i:i+batch_size]
        body = json.dumps({'memory_ids': batch}).encode()
        req = urllib.request.Request(
            f'{BASE_URL}/memories/batch/delete',
            data=body,
            headers={
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json',
            },
            method='POST',
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read())
                deleted += result.get('deleted_count', len(batch))
        except Exception as e:
            print(f'  WARN delete batch: {e}')
            failed += len(batch)
    return deleted, failed


print('Collecting all memories tagged "locomo"...')
all_ids = []
page = 1
while True:
    items, has_more = _list_page('locomo', page)
    for m in items:
        all_ids.append(str(m.id))
    if page % 10 == 0:
        print(f'  page {page}: {len(all_ids)} so far...')
    if not has_more:
        break
    page += 1

print(f'Found {len(all_ids)} locomo memories to delete.')
if not all_ids:
    print('Nothing to delete. Exiting.')
    sys.exit(0)

print(f'Deleting {len(all_ids)} memories...')
deleted, failed = _batch_delete(all_ids)
print(f'Done: {deleted} deleted, {failed} failed.')
