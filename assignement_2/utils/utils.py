import os
from pathlib import Path
from dotenv import load_dotenv


def get__data_dir():
    load_dotenv()
    dir_name = "REPO_DATA_DIR"
    if dir_name in os.environ:
        path = os.getenv(dir_name)
    else:
        return Exception("No data directory found")
    p = Path(path)
    return p
