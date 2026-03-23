import os, json
from pathlib import Path

os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
Path(os.path.expanduser("~/.kaggle/kaggle.json")).write_text(
    json.dumps({"username": "ahbond", "key": "39115a20af8cca72595d6132e2baa2fd"})
)
os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)
os.system("pip install -q 'kaggle>=1.6,<1.7'")
os.makedirs("data", exist_ok=True)
os.system("kaggle competitions download -c nvidia-nemotron-model-reasoning-challenge -p data/")
os.system("cd data && unzip -o '*.zip' 2>/dev/null; ls *.csv")
