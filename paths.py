"""Where things live. The AndroidControl folder can be moved with ANDROIDCONTROL_DIR."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def androidcontrol_dir():
    """Folder holding raw/, images/, a11y/ and episodes.jsonl (made by scripts/get_androidcontrol.sh)."""
    return os.path.abspath(os.environ.get("ANDROIDCONTROL_DIR") or os.path.join(HERE, "data", "androidcontrol"))


def resolve(path):
    """Relative paths are relative to the repository folder."""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(HERE, path))
