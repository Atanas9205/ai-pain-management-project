import sys
from pathlib import Path

def pytest_sessionstart(session):
    """
    Ensure the project root (where 'src/' lives) is importable no matter
    if tests are run via terminal or from inside Jupyter (cwd quirks).
    """
    test_dir = Path(__file__).resolve().parent
    # project root is the parent of 'tests/'
    root = test_dir.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))