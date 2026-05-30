"""
db.py — SQLite persistence for backtest results and rank snapshots.
"""
import json
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "tradeboard.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS backtest_runs (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy   TEXT    NOT NULL DEFAULT 'ml',
                years      INTEGER NOT NULL,
                top_n      INTEGER NOT NULL,
                cost_bps   REAL    NOT NULL,
                result     TEXT    NOT NULL,
                run_at     TEXT,
                created_at TEXT    DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS rank_snapshots (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                as_of      TEXT    NOT NULL,
                result     TEXT    NOT NULL,
                created_at TEXT    DEFAULT (datetime('now'))
            );
        """)
        # migrate: add strategy column if it doesn't exist yet
        cols = [r[1] for r in conn.execute("PRAGMA table_info(backtest_runs)").fetchall()]
        if "strategy" not in cols:
            conn.execute("ALTER TABLE backtest_runs ADD COLUMN strategy TEXT NOT NULL DEFAULT 'ml'")


def save_backtest(years: int, top_n: int, cost_bps: float, result: dict,
                  strategy: str = "ml"):
    with _connect() as conn:
        conn.execute(
            """INSERT INTO backtest_runs
               (strategy, years, top_n, cost_bps, result, run_at)
               VALUES (?,?,?,?,?,?)""",
            (strategy, years, top_n, cost_bps, json.dumps(result), result.get("run_at")),
        )


def get_latest_backtest(years: int = None, top_n: int = None, cost_bps: float = None,
                        strategy: str = "ml") -> dict:
    """Most recent run matching strategy + params, or most recent of that strategy."""
    with _connect() as conn:
        if years is not None and top_n is not None and cost_bps is not None:
            row = conn.execute(
                """SELECT result FROM backtest_runs
                   WHERE strategy=? AND years=? AND top_n=? AND cost_bps=?
                   ORDER BY id DESC LIMIT 1""",
                (strategy, years, top_n, cost_bps),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT result FROM backtest_runs WHERE strategy=? ORDER BY id DESC LIMIT 1",
                (strategy,),
            ).fetchone()
        return json.loads(row["result"]) if row else {}


def save_rank_snapshot(result: dict):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO rank_snapshots (as_of, result) VALUES (?,?)",
            (result.get("as_of", ""), json.dumps(result)),
        )


def get_latest_rank_snapshot() -> dict:
    with _connect() as conn:
        row = conn.execute(
            "SELECT result FROM rank_snapshots ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return json.loads(row["result"]) if row else {}


def list_backtest_runs(strategy: str = None) -> list[dict]:
    """Summary of all runs — no result blob. Optionally filter by strategy."""
    with _connect() as conn:
        if strategy:
            rows = conn.execute(
                """SELECT id, strategy, years, top_n, cost_bps, run_at, created_at
                   FROM backtest_runs WHERE strategy=? ORDER BY id DESC""",
                (strategy,),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT id, strategy, years, top_n, cost_bps, run_at, created_at
                   FROM backtest_runs ORDER BY id DESC"""
            ).fetchall()
        return [dict(r) for r in rows]


init_db()
