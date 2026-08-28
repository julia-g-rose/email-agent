"""Enron email search environment: data models, SQLite DB, search/read tools, scenarios.

Faithful port of the ART·E example. Downloads the Enron email dataset from Hugging
Face (`corbt/enron-emails`) into a local SQLite DB with an FTS5 index, and loads the
paired question/answer scenarios (`corbt/enron_emails_sample_questions`).
"""

from __future__ import annotations

import os
import random
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import List, Literal, Optional

from datasets import Dataset, Features, Sequence, Value, load_dataset
from pydantic import BaseModel
from tqdm import tqdm


# ---- data models ----------------------------------------------------------
class Email(BaseModel):
    message_id: str
    date: str  # ISO 8601 string 'YYYY-MM-DD HH:MM:SS'
    subject: Optional[str] = None
    from_address: Optional[str] = None
    to_addresses: List[str] = []
    cc_addresses: List[str] = []
    bcc_addresses: List[str] = []
    body: Optional[str] = None
    file_name: Optional[str] = None


class Scenario(BaseModel):
    id: int
    question: str
    answer: str
    message_ids: List[str]
    how_realistic: float
    inbox_address: str
    query_date: str
    split: Literal["train", "test"]


@dataclass
class SearchResult:
    message_id: str
    snippet: str


class FinalAnswer(BaseModel):
    answer: str
    source_ids: list[str]


# ---- database -------------------------------------------------------------
DB_PATH = os.environ.get("EMAIL_DB_PATH", "./enron_emails.db")
EMAIL_DATASET_REPO_ID = "corbt/enron-emails"
SCENARIO_DATASET_REPO_ID = "corbt/enron_emails_sample_questions"

db_conn: Optional[sqlite3.Connection] = None


def create_email_database() -> sqlite3.Connection:
    """Create the email database from the Hugging Face dataset."""
    print("Creating email database from Hugging Face dataset...")
    print("Downloading and processing the full Enron email dataset — this may take several minutes...")

    SQL_CREATE_TABLES = """
    DROP TABLE IF EXISTS recipients;
    DROP TABLE IF EXISTS emails_fts;
    DROP TABLE IF EXISTS emails;

    CREATE TABLE emails (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message_id TEXT UNIQUE,
        subject TEXT,
        from_address TEXT,
        date TEXT,
        body TEXT,
        file_name TEXT
    );

    CREATE TABLE recipients (
        email_id TEXT,
        recipient_address TEXT,
        recipient_type TEXT
    );
    """

    SQL_CREATE_INDEXES_TRIGGERS = """
    CREATE INDEX idx_emails_from ON emails(from_address);
    CREATE INDEX idx_emails_date ON emails(date);
    CREATE INDEX idx_emails_message_id ON emails(message_id);
    CREATE INDEX idx_recipients_address ON recipients(recipient_address);
    CREATE INDEX idx_recipients_type ON recipients(recipient_type);
    CREATE INDEX idx_recipients_email_id ON recipients(email_id);
    CREATE INDEX idx_recipients_address_email ON recipients(recipient_address, email_id);

    CREATE VIRTUAL TABLE emails_fts USING fts5(
        subject,
        body,
        content='emails',
        content_rowid='id'
    );

    CREATE TRIGGER emails_ai AFTER INSERT ON emails BEGIN
        INSERT INTO emails_fts (rowid, subject, body)
        VALUES (new.id, new.subject, new.body);
    END;

    CREATE TRIGGER emails_ad AFTER DELETE ON emails BEGIN
        DELETE FROM emails_fts WHERE rowid=old.id;
    END;

    CREATE TRIGGER emails_au AFTER UPDATE ON emails BEGIN
        UPDATE emails_fts SET subject=new.subject, body=new.body WHERE rowid=old.id;
    END;
    """

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.executescript(SQL_CREATE_TABLES)
    conn.commit()

    print("Loading full email dataset...")
    expected_features = Features(
        {
            "message_id": Value("string"),
            "subject": Value("string"),
            "from": Value("string"),
            "to": Sequence(Value("string")),
            "cc": Sequence(Value("string")),
            "bcc": Sequence(Value("string")),
            "date": Value("timestamp[us]"),
            "body": Value("string"),
            "file_name": Value("string"),
        }
    )

    dataset = load_dataset(EMAIL_DATASET_REPO_ID, features=expected_features, split="train")
    print(f"Dataset contains {len(dataset)} total emails")

    print("Populating database with all emails...")
    conn.execute("PRAGMA synchronous = OFF;")
    conn.execute("PRAGMA journal_mode = MEMORY;")
    conn.execute("BEGIN TRANSACTION;")

    record_count = 0
    skipped_count = 0
    duplicate_count = 0
    processed_emails: set = set()

    for email_data in tqdm(dataset, desc="Inserting emails"):
        message_id = email_data["message_id"]
        subject = email_data["subject"]
        from_address = email_data["from"]
        date_obj: datetime = email_data["date"]
        body = email_data["body"]
        file_name = email_data["file_name"]
        to_list = [str(addr) for addr in email_data["to"] if addr]
        cc_list = [str(addr) for addr in email_data["cc"] if addr]
        bcc_list = [str(addr) for addr in email_data["bcc"] if addr]

        total_recipients = len(to_list) + len(cc_list) + len(bcc_list)
        if len(body) > 5000:
            skipped_count += 1
            continue
        if total_recipients > 30:
            skipped_count += 1
            continue

        email_key = (subject, body, from_address)
        if email_key in processed_emails:
            duplicate_count += 1
            continue
        processed_emails.add(email_key)

        date_str = date_obj.strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "INSERT INTO emails (message_id, subject, from_address, date, body, file_name) VALUES (?, ?, ?, ?, ?, ?)",
            (message_id, subject, from_address, date_str, body, file_name),
        )

        recipient_data = (
            [(message_id, a, "to") for a in to_list]
            + [(message_id, a, "cc") for a in cc_list]
            + [(message_id, a, "bcc") for a in bcc_list]
        )
        if recipient_data:
            cursor.executemany(
                "INSERT INTO recipients (email_id, recipient_address, recipient_type) VALUES (?, ?, ?)",
                recipient_data,
            )
        record_count += 1

    conn.commit()
    print("Creating indexes and FTS...")
    cursor.executescript(SQL_CREATE_INDEXES_TRIGGERS)
    cursor.execute('INSERT INTO emails_fts(emails_fts) VALUES("rebuild")')
    conn.commit()

    print(f"Successfully created database with {record_count} emails.")
    print(f"Skipped {skipped_count} due to length/recipient limits; {duplicate_count} duplicates.")
    return conn


def get_db_connection() -> sqlite3.Connection:
    global db_conn
    if db_conn is None:
        if os.path.exists(DB_PATH):
            print(f"Loading existing database from {DB_PATH}")
            db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        else:
            db_conn = create_email_database()
    return db_conn


# ---- tools ----------------------------------------------------------------
def search_emails(
    inbox: str,
    keywords: List[str],
    from_addr: Optional[str] = None,
    to_addr: Optional[str] = None,
    sent_after: Optional[str] = None,
    sent_before: Optional[str] = None,
    max_results: int = 10,
) -> List[SearchResult]:
    """Search the email database by keywords and filters (FTS5, AND semantics)."""
    conn = get_db_connection()
    cursor = conn.cursor()

    where_clauses: List[str] = []
    params: List = []

    if not keywords:
        raise ValueError("No keywords provided for search.")
    if max_results > 10:
        raise ValueError("max_results must be less than or equal to 10.")

    fts_query = " ".join(f""" "{k.replace('"', '""')}" """ for k in keywords)
    where_clauses.append("fts.emails_fts MATCH ?")
    params.append(fts_query)

    where_clauses.append(
        """
        (e.from_address = ? OR EXISTS (
            SELECT 1 FROM recipients r_inbox
            WHERE r_inbox.recipient_address = ? AND r_inbox.email_id = e.message_id
        ))
        """
    )
    params.extend([inbox, inbox])

    if from_addr:
        where_clauses.append("e.from_address = ?")
        params.append(from_addr)
    if to_addr:
        where_clauses.append(
            "EXISTS (SELECT 1 FROM recipients r_to WHERE r_to.recipient_address = ? AND r_to.email_id = e.message_id)"
        )
        params.append(to_addr)
    if sent_after:
        where_clauses.append("e.date >= ?")
        params.append(f"{sent_after} 00:00:00")
    if sent_before:
        where_clauses.append("e.date < ?")
        params.append(f"{sent_before} 00:00:00")

    sql = f"""
        SELECT e.message_id, snippet(emails_fts, -1, '<b>', '</b>', ' ... ', 15) as snippet
        FROM emails e JOIN emails_fts fts ON e.id = fts.rowid
        WHERE {" AND ".join(where_clauses)}
        ORDER BY e.date DESC
        LIMIT ?;
    """
    params.append(max_results)
    cursor.execute(sql, params)
    return [SearchResult(message_id=row[0], snippet=row[1]) for row in cursor.fetchall()]


def read_email(message_id: str) -> Optional[Email]:
    """Retrieve a single email by its message_id."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT message_id, date, subject, from_address, body, file_name FROM emails WHERE message_id = ?",
        (message_id,),
    )
    email_row = cursor.fetchone()
    if not email_row:
        return None
    msg_id, date, subject, from_addr, body, file_name = email_row

    cursor.execute(
        "SELECT recipient_address, recipient_type FROM recipients WHERE email_id = ?",
        (message_id,),
    )
    to_addresses, cc_addresses, bcc_addresses = [], [], []
    for addr, type_val in cursor.fetchall():
        {"to": to_addresses, "cc": cc_addresses, "bcc": bcc_addresses}.get(
            type_val.lower(), []
        ).append(addr)

    return Email(
        message_id=msg_id,
        date=date,
        subject=subject,
        from_address=from_addr,
        to_addresses=to_addresses,
        cc_addresses=cc_addresses,
        bcc_addresses=bcc_addresses,
        body=body,
        file_name=file_name,
    )


# ---- scenarios ------------------------------------------------------------
def load_scenarios(
    split: Literal["train", "test"] = "train",
    limit: Optional[int] = None,
    max_messages: Optional[int] = 1,
    shuffle: bool = False,
    seed: Optional[int] = None,
) -> List[Scenario]:
    """Load question/answer scenarios from the Hugging Face dataset."""
    print(f"Loading {split} scenarios from Hugging Face...")
    dataset: Dataset = load_dataset(SCENARIO_DATASET_REPO_ID, split=split)

    if max_messages is not None:
        dataset = dataset.filter(lambda x: len(x["message_ids"]) <= max_messages)
    if shuffle or (seed is not None):
        dataset = dataset.shuffle(seed=seed) if seed is not None else dataset.shuffle()

    scenarios = [Scenario(**row, split=split) for row in dataset]
    if max_messages is not None:
        scenarios = [s for s in scenarios if len(s.message_ids) <= max_messages]
    if shuffle:
        (random.Random(seed) if seed is not None else random).shuffle(scenarios)
    if limit is not None:
        scenarios = scenarios[:limit]

    print(f"Loaded {len(scenarios)} scenarios.")
    return scenarios
