"""Скачивание обученных моделей (архив из GitHub Release) → models/.

  python scripts/download_models.py https://github.com/<user>/ugc-moderation/releases/download/v1.0.0/models.tar.gz
"""
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    if len(sys.argv) < 2 or "<" in sys.argv[1]:
        sys.exit("Укажите URL архива: make download-models MODELS_URL=https://.../models.tar.gz")
    url = sys.argv[1]
    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp:
        print(f"скачиваю {url}")
        urllib.request.urlretrieve(url, tmp.name)
        with tarfile.open(tmp.name) as tar:
            for member in tar.getmembers():  # защита от путей вида ../../
                if not (ROOT / member.name).resolve().is_relative_to(ROOT / "models"):
                    sys.exit(f"подозрительный путь в архиве: {member.name}")
            tar.extractall(ROOT)
    print("модели распакованы в models/")


if __name__ == "__main__":
    main()
