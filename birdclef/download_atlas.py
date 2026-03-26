"""Download BirdCLEF 2026 data using Bearer auth."""
import requests
import os

TOKEN = "KGAT_4426b9d6e4d1a86800213ca9b8c74212"
URL = "https://www.kaggle.com/api/v1/competitions/data/download-all/birdclef-2026"
OUT = "birdclef-2026.zip"

print("Downloading BirdCLEF 2026...")
r = requests.get(URL, headers={"Authorization": f"Bearer {TOKEN}"}, stream=True)
print(f"Status: {r.status_code}")
total = int(r.headers.get("content-length", 0))
print(f"Size: {total / 1e9:.1f} GB")

dl = 0
with open(OUT, "wb") as f:
    for chunk in r.iter_content(8 * 1024 * 1024):
        f.write(chunk)
        dl += len(chunk)
        if dl % (500 * 1024 * 1024) < 8 * 1024 * 1024:
            pct = 100 * dl / max(total, 1)
            print(f"  {dl / 1e9:.1f}/{total / 1e9:.1f} GB ({pct:.0f}%)")

print(f"Done: {os.path.getsize(OUT) / 1e9:.1f} GB")
