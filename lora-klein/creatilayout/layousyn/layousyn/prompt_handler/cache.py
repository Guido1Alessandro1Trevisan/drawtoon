import json
import sqlite3
from typing import Any, Dict, List


class KeyObjectCache:
    """
    Store key: object in SQLite database
    Object can be any dictionary, list, etc. which can be serialized to JSON
    """

    def __init__(self, db_path: str = "cache/key_object.db") -> None:
        self.db_path: str = db_path

        # connect to db
        try:
            self.conn: sqlite3.Connection = sqlite3.connect(self.db_path)
        except sqlite3.Error as e:
            self.conn = sqlite3.connect(":memory:")
        self.cursor: sqlite3.Cursor = self.conn.cursor()

        # create db if not exists
        self.create_table()

    def close(self) -> None:
        self.conn.close()

    def create_table(self) -> None:
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS key_object (
                key TEXT PRIMARY KEY,
                object TEXT
            )
            """
        )

    def insert(self, key: str, object: Any) -> None:
        self.cursor.execute(
            "INSERT OR REPLACE INTO key_object (key, object) VALUES (?, ?)",
            (key, json.dumps(object)),
        )
        self.conn.commit()

    def get_key_object_map(self, keys: List[str]) -> Dict[str, Dict[str, int]]:
        self.cursor.execute(
            "SELECT key, object FROM key_object WHERE key IN ({})".format(
                ",".join("?" * len(keys))
            ),
            keys,
        )
        output = self.cursor.fetchall()
        output = {item[0]: json.loads(item[1]) for item in output}
        return output
