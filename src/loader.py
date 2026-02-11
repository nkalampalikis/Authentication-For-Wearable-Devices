"""
Database loader utility for processing raw sensor data into SQLite databases.

Run this module directly to initialize databases for both BVP and BCG:
    python -m src.loader
"""

import os
import os.path
import sqlite3

from .db import DB
from .signal_filter import SignalType


def get_project_root():
    """Get the project root directory."""
    src_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.dirname(src_dir)


def load_db(window_size, signal_type=SignalType.BCG, project_root=None):
    """
    Load raw sensor data into a SQLite database.

    Args:
        window_size: Size of the time window in seconds
        signal_type: Type of signal processing (SignalType.BVP or SignalType.BCG)
        project_root: Optional project root path. If None, auto-detected.

    Returns:
        int: Number of sequences successfully loaded
    """
    if project_root is None:
        project_root = get_project_root()

    # Create signal-specific database directory
    db_dir_name = f"databases_{signal_type.value.upper()}"
    db_dir = os.path.join(project_root, db_dir_name)
    data_dir = os.path.join(project_root, "data")

    if not os.path.isdir(db_dir):
        print(f"Creating {db_dir_name} Directory")
        os.mkdir(db_dir)

    db_path = os.path.join(db_dir, f"db_{window_size}.sqlite")

    # Check if database already exists and has data
    if os.path.isfile(db_path):
        try:
            existing_db = DB(db_path, init=False)
            count = existing_db.cursor.execute("SELECT COUNT(*) FROM data").fetchone()[0]
            if count > 0:
                print(f"  Database already exists with {count} sequences. Skipping.")
                print(f"  (Delete {db_path} to recreate)")
                return 0
        except sqlite3.OperationalError:
            # Table doesn't exist, proceed with initialization
            pass

    db = DB(db_path, init=True)

    targets = list(range(1, 29))
    loaded_count = 0
    skipped_count = 0

    for i in targets:
        for j in range(1, 5):  # Sessions
            for k in range(1, 6):  # Sequences
                g_file = os.path.join(data_dir, str(i), str(j), str(k), "gyro.csv")
                a_file = os.path.join(data_dir, str(i), str(j), str(k), "accel.csv")

                if os.path.isfile(g_file) and os.path.isfile(a_file):
                    table_name = f"{i}_{j}_{k}_{window_size}"

                    # Check if this table already exists
                    existing = db.cursor.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                        [table_name]
                    ).fetchone()

                    if existing:
                        skipped_count += 1
                        continue

                    print(f"Loading {i}-{j}-{k}")
                    try:
                        db.load_from_csv(g_file, a_file, str(i), j, k, window_size, 50, signal_type)
                        loaded_count += 1
                    except sqlite3.OperationalError as e:
                        if "already exists" in str(e):
                            print(f"  Table {table_name} already exists, skipping")
                            skipped_count += 1
                        else:
                            raise

    if skipped_count > 0:
        print(f"  Skipped {skipped_count} existing sequences")

    return loaded_count


def load_all_databases(signal_type=None):
    """
    Load databases for all standard window sizes (2-5 seconds).

    Args:
        signal_type: Type of signal processing. If None, loads both BVP and BCG.
    """
    signal_types = [signal_type] if signal_type else [SignalType.BCG, SignalType.BVP]

    for sig_type in signal_types:
        print(f"\n{'='*60}")
        print(f"Processing {sig_type.value.upper()} Signal Type")
        print(f"{'='*60}")

        for window_size in [2, 3, 4, 5]:
            print(f"\n=== Loading window size {window_size} ({sig_type.value.upper()}) ===")
            loaded = load_db(window_size, sig_type)
            if loaded > 0:
                print(f"  Loaded {loaded} sequences")


if __name__ == "__main__":
    print("Initializing databases for both BCG and BVP signal types...")
    load_all_databases()
    print("\nDatabase initialization complete!")
