from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent / "Datos"


def load(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.is_absolute():
        if path.parts and path.parts[0] == DATA_DIR.name:
            path = DATA_DIR.parent / path
        else:
            path = DATA_DIR / path

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    return pd.read_csv(path)