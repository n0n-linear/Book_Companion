"""
Book Companion+ (v4.6 — Great Expectations)
Persistent DB • FTS fixes • Antagonist • Library multi‑select • Robust EPUB import • Version codenames

Run:
  pip install -r requirements.txt
  streamlit run app_v4_6_great_expectations.py
"""
from __future__ import annotations
import os, io, re, json, time, sqlite3, hashlib, platform, shutil, zipfile as _zipfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import Counter

import streamlit as st

try:
    import matplotlib.pyplot as plt
    MPL_OK = True
except Exception:
    MPL_OK = False

try:
    from openai import OpenAI
    _OPENAI_OK = True
except Exception:
    _OPENAI_OK = False

from ebooklib import epub
from bs4 import BeautifulSoup

def _prefs_get(key:str, default:Optional[str]=None) -> Optional[str]:
    try:
        con = get_db()
        row = con.execute("SELECT value_json FROM user_prefs WHERE key=?", (key,)).fetchone()
        if not row:
            return default
        import json as _json
        return _json.loads(row["value_json"])
    except Exception:
        return default

def _prefs_set(key:str, value:str) -> None:
    try:
        con = get_db()
        import json as _json
        with con:
            con.execute("INSERT INTO user_prefs(key,value_json) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json",
                        (key, _json.dumps(value)))
    except Exception:
        pass

def get_api_key() -> Optional[str]:
    # Priority: session -> env -> user_prefs -> st.secrets (if present)
    import os as _os
    import streamlit as _st

def _row_get(row, key, default=None):
    try:
        # sqlite3.Row supports key access; default if missing/None
        if hasattr(row, 'keys') and key in row.keys():
            val = row[key]
        else:
            # fall back to dict-like get if available
            val = row.get(key) if hasattr(row, 'get') else None
        return default if val is None else val
    except Exception:
        return default


    if "OPENAI_API_KEY" in _st.session_state and _st.session_state["OPENAI_API_KEY"]:
        return _st.session_state["OPENAI_API_KEY"]
    if _os.environ.get("OPENAI_API_KEY"):
        return _os.environ.get("OPENAI_API_KEY")
    val = _prefs_get("OPENAI_API_KEY")
    if val:
        # Cache into session/env for this run
        _st.session_state["OPENAI_API_KEY"] = val
        _os.environ["OPENAI_API_KEY"] = val
        return val
    try:
        sec = _st.secrets.get("OPENAI_API_KEY", None)  # type: ignore[attr-defined]
        if sec:
            _st.session_state["OPENAI_API_KEY"] = sec
            _os.environ["OPENAI_API_KEY"] = sec
            return sec
    except Exception:
        pass
    return None




def _prefs_get_json(key:str, default=None):
    try:
        con = get_db()
        row = con.execute("SELECT value_json FROM user_prefs WHERE key=?", (key,)).fetchone()
        if not row:
            return default
        import json as _json
        return _json.loads(row["value_json"])
    except Exception:
        return default

def _prefs_set_json(key:str, obj) -> None:
    try:
        con = get_db()
        import json as _json
        with con:
            con.execute("INSERT INTO user_prefs(key,value_json) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json",
                        (key, _json.dumps(obj)))
    except Exception:
        pass

def get_model_prefs():
    defaults = {
        "provider": "OpenAI",
        "base_url": "",
        "model": os.environ.get("OPENAI_BOOK_MODEL", "gpt-4o-mini"),
        "test_mode": False
    }
    saved = _prefs_get_json("MODEL_PREFS", {}) or {}
    for k,v in defaults.items():
        saved.setdefault(k, v)
    return saved



def _fts_query_from_text(text:str, max_terms:int=12) -> Optional[str]:
    """
    Convert arbitrary user text into a safe FTS5 query string.
    - Extract alphanumeric tokens (keeps simple apostrophes inside words).
    - Join with OR to broaden recall.
    - Return None if no valid tokens found.
    """
    if not text:
        return None
    toks = re.findall(r"[A-Za-z0-9']+", text.lower())
    toks = [t.strip("'") for t in toks if t.strip("'")]
    toks = toks[:max_terms]
    if not toks:
        return None
    return " OR ".join(toks)


# ------------------------ Versioning with classic-book codename ------------------------
VERSION = "4.17"
CODENAME = os.environ.get("BOOK_CODENAME", "Middlemarch")
VERSION_DISPLAY = f"v{VERSION} — {CODENAME}"

# ------------------------ Data directory (persistent) ------------------------
def _default_data_dir() -> Path:
    sys = platform.system()
    if sys == "Darwin":
        return Path.home() / "Library" / "Application Support" / "BookCompanion"
    elif sys == "Windows":
        return Path(os.environ.get("APPDATA", str(Path.home()))) / "BookCompanion"
    else:
        return Path.home() / ".local" / "share" / "book_companion"

OLD_DATA_DIR = Path("./data")
DATA_DIR = _default_data_dir()
FILES_DIR = DATA_DIR / "files"
DB_PATH = DATA_DIR / "reader.db"
FILES_DIR.mkdir(parents=True, exist_ok=True)

# One‑time migration from ./data to OS‑stable directory
old_db = OLD_DATA_DIR / "reader.db"
if old_db.exists() and not DB_PATH.exists():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(old_db, DB_PATH)
    old_files = OLD_DATA_DIR / "files"
    if old_files.exists():
        for p in old_files.glob("*"):
            target = FILES_DIR / p.name
            if not target.exists():
                shutil.copy2(p, target)

MODEL_DEFAULT = os.environ.get("OPENAI_BOOK_MODEL", "gpt-5-mini")
TOPK_DEFAULT = int(os.environ.get("BOOK_TOPK", "12"))
CHUNK_CHAR_BUDGET = int(os.environ.get("BOOK_CHUNK_CHARS", "4800"))

def get_db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA journal_mode = WAL")
    con.execute("PRAGMA synchronous = NORMAL")
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS books(
            id INTEGER PRIMARY KEY, title TEXT, author TEXT, ext TEXT, path TEXT, added_at TEXT
        );

        CREATE TABLE IF NOT EXISTS book_pages(
            id INTEGER PRIMARY KEY, book_id INTEGER, page_no INTEGER, chapter TEXT, text TEXT,
            UNIQUE(book_id,page_no)
        );
        CREATE INDEX IF NOT EXISTS idx_book_pages_book_page ON book_pages(book_id, page_no);

        CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(
            text, page_no UNINDEXED, book_id UNINDEXED, chapter UNINDEXED,
            content='book_pages', content_rowid='id'
        );
        CREATE TRIGGER IF NOT EXISTS book_pages_ai AFTER INSERT ON book_pages BEGIN
            INSERT INTO pages_fts(rowid, text, page_no, book_id, chapter)
            VALUES (new.id, new.text, new.page_no, new.book_id, new.chapter);
        END;
        CREATE TRIGGER IF NOT EXISTS book_pages_ad AFTER DELETE ON book_pages BEGIN
            INSERT INTO pages_fts(pages_fts, rowid, text, page_no, book_id, chapter)
            VALUES ('delete', old.id, old.text, old.page_no, old.book_id, old.chapter);
        END;
        CREATE TRIGGER IF NOT EXISTS book_pages_au AFTER UPDATE ON book_pages BEGIN
            INSERT INTO pages_fts(pages_fts, rowid, text, page_no, book_id, chapter)
            VALUES ('delete', old.id, old.text, old.page_no, old.book_id, old.chapter);
            INSERT INTO pages_fts(rowid, text, page_no, book_id, chapter)
            VALUES (new.id, new.text, new.page_no, new.book_id, new.chapter);
        END;

        CREATE TABLE IF NOT EXISTS threads(
            id INTEGER PRIMARY KEY, name TEXT, created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS thread_books(
            thread_id INTEGER, book_id INTEGER, PRIMARY KEY(thread_id,book_id)
        );
        CREATE INDEX IF NOT EXISTS idx_thread_books_thread ON thread_books(thread_id);
        CREATE INDEX IF NOT EXISTS idx_thread_books_book ON thread_books(book_id);

        CREATE TABLE IF NOT EXISTS messages(
            id INTEGER PRIMARY KEY, thread_id INTEGER, role TEXT, content TEXT, created_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id);

        CREATE TABLE IF NOT EXISTS annotations(
            id INTEGER PRIMARY KEY, book_id INTEGER, page_no INTEGER,
            category TEXT CHECK(category in ('theme','topic','character','timeline','language','style','quote')),
            label TEXT, details TEXT, created_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_annotations_book ON annotations(book_id);

        CREATE VIRTUAL TABLE IF NOT EXISTS annotations_fts USING fts5(
            label, details, category UNINDEXED, book_id UNINDEXED, page_no UNINDEXED,
            content='annotations', content_rowid='id'
        );
        CREATE TRIGGER IF NOT EXISTS annotations_ai AFTER INSERT ON annotations BEGIN
            INSERT INTO annotations_fts(rowid, label, details, category, book_id, page_no)
            VALUES (new.id, new.label, new.details, new.category, new.book_id, new.page_no);
        END;
        CREATE TRIGGER IF NOT EXISTS annotations_ad AFTER DELETE ON annotations BEGIN
            INSERT INTO annotations_fts(annotations_fts, rowid, label, details, category, book_id, page_no)
            VALUES ('delete', old.id, old.label, old.details, old.category, old.book_id, old.page_no);
        END;
        CREATE TRIGGER IF NOT EXISTS annotations_au AFTER UPDATE ON annotations BEGIN
            INSERT INTO annotations_fts(annotations_fts, rowid, label, details, category, book_id, page_no)
            VALUES ('delete', old.id, old.label, old.details, old.category, old.book_id, old.page_no);
            INSERT INTO annotations_fts(rowid, label, details, category, book_id, page_no)
            VALUES (new.id, new.label, new.details, new.category, new.book_id, new.page_no);
        END;

        CREATE TABLE IF NOT EXISTS characters(
            id INTEGER PRIMARY KEY, book_id INTEGER, name TEXT, aliases TEXT,
            first_page INTEGER, description TEXT, created_at TEXT
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS characters_fts USING fts5(
            name, aliases, description, book_id UNINDEXED, first_page UNINDEXED,
            content='characters', content_rowid='id'
        );
        CREATE TRIGGER IF NOT EXISTS characters_ai AFTER INSERT ON characters BEGIN
            INSERT INTO characters_fts(rowid, name, aliases, description, book_id, first_page)
            VALUES (new.id, new.name, new.aliases, new.description, new.book_id, new.first_page);
        END;
        CREATE TRIGGER IF NOT EXISTS characters_ad AFTER DELETE ON characters BEGIN
            INSERT INTO characters_fts(characters_fts, rowid, name, aliases, description, book_id, first_page)
            VALUES ('delete', old.id, old.name, old.aliases, old.description, old.book_id, old.first_page);
        END;
        CREATE TRIGGER IF NOT EXISTS characters_au AFTER UPDATE ON characters BEGIN
            INSERT INTO characters_fts(characters_fts, rowid, name, aliases, description, book_id, first_page)
            VALUES ('delete', old.id, old.name, old.aliases, old.description, old.book_id, old.first_page);
            INSERT INTO characters_fts(rowid, name, aliases, description, book_id, first_page)
            VALUES (new.id, new.name, new.aliases, new.description, new.book_id, new.first_page);
        END;

        CREATE TABLE IF NOT EXISTS study_notes(
            id INTEGER PRIMARY KEY, thread_id INTEGER, title TEXT, body TEXT, tags_json TEXT, created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS study_summaries(
            id INTEGER PRIMARY KEY, book_id INTEGER, title TEXT, summary TEXT, created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS events(
            id INTEGER PRIMARY KEY, ts TEXT, event TEXT, book_id INTEGER, thread_id INTEGER, page_no INTEGER, data_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_events_book ON events(book_id);
        CREATE INDEX IF NOT EXISTS idx_events_thread ON events(thread_id);

        CREATE TABLE IF NOT EXISTS user_prefs(
            key TEXT PRIMARY KEY, value_json TEXT
        );
        """
    )
    return con

def _book_title(book_id: int) -> str:
    """Return the book title by id; fallback to 'Book <id>' if missing."""
    try:
        con = get_db()
        row = con.execute("SELECT title FROM books WHERE id=?", (book_id,)).fetchone()
        title = None
        if row is not None:
            try:
                title = row['title'] if hasattr(row, '__getitem__') else None
            except Exception:
                title = None
        return (title or f"Book {book_id}")
    except Exception:
        return f"Book {book_id}"


def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat()+"Z"

def file_hash(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]

def clean_html(html:str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    [tag.decompose() for tag in soup(["script","style"])]
    return soup.get_text(" ")

def parse_txt(raw:bytes):
    t = raw.decode("utf-8", errors="ignore")
    page = 4800
    return [(i+1, t[i:i+page], None) for i in range(0, len(t), page)]

def parse_epub(raw:bytes):
    """
    Robust EPUB parser with tolerant ebooklib and ZIP fallback.
    """
    import tempfile, os as _os
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".epub")
    try:
        tmp.write(raw); tmp.flush(); tmp.close()
        # Attempt 1: tolerant read_epub
        try:
            book = epub.read_epub(tmp.name, options={"ignore_ncx": True, "ignore_toc": True})
            title = (book.get_metadata('DC','title') or [["Untitled"]])[0][0]
            author = (book.get_metadata('DC','creator') or [[None]])[0][0]

            # Build TOC map if any
            href_title = {}
            try:
                def _walk(nodes, trail=None):
                    trail = trail or []
                    for n in nodes:
                        if isinstance(n, epub.Link):
                            t2 = " ".join([*trail, n.title]).strip()
                            href_title[n.href.split('#')[0]] = t2
                        elif isinstance(n, tuple) and len(n) >= 2:
                            node, children = n[0], n[1]
                            t = node.title if hasattr(node, 'title') else str(node)
                            _walk(children, [*trail, t])
                _walk(getattr(book, "toc", []))
            except Exception:
                pass

            pages, page_chars, page_no = [], 4800, 1
            for item in book.get_items():
                if item.get_type() == epub.ITEM_DOCUMENT:
                    raw_html = item.get_content().decode("utf-8", errors="ignore")
                    text = clean_html(raw_html)
                    href = item.get_name() or item.get_id() or ""
                    chap = None
                    for key, tlabel in href_title.items():
                        if key in href:
                            chap = tlabel; break
                    for i in range(0, len(text), page_chars):
                        chunk = text[i:i+page_chars].strip()
                        if chunk:
                            pages.append((page_no, chunk, chap)); page_no += 1
            return title or "Untitled", author, pages
        except Exception:
            # Attempt 2: manual ZIP fallback
            with _zipfile.ZipFile(tmp.name, "r") as zf:
                names = [n for n in zf.namelist() if n.lower().endswith((".xhtml",".html",".htm"))]
                def sort_key(n):
                    nl = n.lower(); prio = 0
                    if "oebps" in nl: prio -= 10
                    if "content" in nl: prio -= 5
                    return (prio, n)
                names.sort(key=sort_key)
                chunks = []
                for n in names:
                    try:
                        data = zf.read(n)
                        try:
                            txt = data.decode("utf-8", errors="ignore")
                        except Exception:
                            txt = data.decode("latin-1", errors="ignore")
                        chunks.append(clean_html(txt))
                    except KeyError:
                        continue
                big = "\\n\\n".join(chunks)
                title = "Untitled (fallback)"
                author = None
                pages, page_chars = [], 4800
                for i in range(0, len(big), page_chars):
                    piece = big[i:i+page_chars].strip()
                    if piece:
                        pages.append((len(pages)+1, piece, None))
                return title, author, pages
    finally:
        try:
            _os.unlink(tmp.name)
        except Exception:
            pass

def parse_mobi(raw:bytes):
    try:
        import mobi
    except Exception:
        return ("MOBI (untitled)", None, parse_txt(raw))
    out = DATA_DIR / "_mobi_tmp"
    out.mkdir(exist_ok=True)
    tmp = out / f"{time.time_ns()}.mobi"
    tmp.write_bytes(raw)
    m = mobi.Mobi(tmp); m.process()
    html = next(p for p in out.glob("*.html") if p.stat().st_size > 0).read_text(encoding="utf-8", errors="ignore")
    txt = clean_html(html)
    pages, pno = [], 1
    for i in range(0, len(txt), 4800):
        chunk = txt[i:i+4800].strip()
        if chunk:
            pages.append((pno, chunk, None)); pno += 1
    return ("MOBI (converted)", None, pages)

def add_book(filename:str, ext:str, raw:bytes) -> int:
    con = get_db()
    if ext == ".txt":
        title, author = Path(filename).stem, None; pages = parse_txt(raw)
    elif ext == ".epub":
        title, author, pages = parse_epub(raw)
    elif ext == ".mobi":
        title, author, pages = parse_mobi(raw)
    else:
        raise ValueError("Unsupported format: " + ext)
    fid = file_hash(raw); path = FILES_DIR / f"{fid}{ext}"
    if not path.exists(): path.write_bytes(raw)
    with con:
        cur = con.execute("INSERT INTO books(title,author,ext,path,added_at) VALUES(?,?,?,?,?)",
                          (title, author, ext, str(path), now_iso()))
        bid = cur.lastrowid
        con.executemany("INSERT INTO book_pages(book_id,page_no,chapter,text) VALUES(?,?,?,?)",
                        [(bid, p, chap, txt) for (p, txt, chap) in pages])
        con.execute("INSERT INTO events(ts,event,book_id,data_json) VALUES(?,?,?,?)",
                    (now_iso(), "book_added", bid, json.dumps({"title": title})))
    return int(bid)

def list_books():
    return list(get_db().execute("SELECT * FROM books ORDER BY added_at DESC"))


def _book_file_exists(p:str) -> bool:
    try:
        return Path(p).exists()
    except Exception:
        return False

def cleanup_invalid_books() -> dict:
    """Remove books with missing paths/files or zero pages; prune orphaned pages/fts."""
    con = sqlite3.connect(DB_PATH); con.row_factory = sqlite3.Row
    removed = {"books":0,"pages":0,"orphans":0}
    with con:
        # Books with missing path or file on disk
        bad = list(con.execute("SELECT id, path FROM books WHERE path IS NULL OR TRIM(path)=''"))
        for b in bad:
            con.execute("DELETE FROM book_pages WHERE book_id=?", (b["id"],))
            con.execute("DELETE FROM books WHERE id=?", (b["id"],))
            removed["books"] += 1
        # Books whose file no longer exists
        missing = list(con.execute("SELECT id, path FROM books WHERE path IS NOT NULL AND TRIM(path)<>''"))
        for b in missing:
            if not _book_file_exists(b["path"]):
                con.execute("DELETE FROM book_pages WHERE book_id=?", (b["id"],))
                con.execute("DELETE FROM books WHERE id=?", (b["id"],))
                removed["books"] += 1
        # Books with zero pages
        zero = list(con.execute("SELECT b.id FROM books b LEFT JOIN book_pages p ON p.book_id=b.id GROUP BY b.id HAVING COUNT(p.id)=0"))
        for z in zero:
            con.execute("DELETE FROM books WHERE id=?", (z["id"],))
            removed["books"] += 1
        # Orphaned pages
        orphans = list(con.execute("SELECT p.id FROM book_pages p LEFT JOIN books b ON b.id=p.book_id WHERE b.id IS NULL"))
        if orphans:
            con.executemany("DELETE FROM book_pages WHERE id=?", [(o["id"],) for o in orphans])
            removed["orphans"] += len(orphans)
        # Vacuum FTS content table via update trick (no-op if empty)
        con.execute("DELETE FROM pages_fts WHERE rowid NOT IN (SELECT id FROM book_pages)")
    return removed

def list_chapters(book_id:int):
    con = get_db()
    return list(con.execute(
        "SELECT COALESCE(chapter,'Unknown chapter') AS chapter, MIN(page_no) AS start_page, MAX(page_no) AS end_page "
        "FROM book_pages WHERE book_id=? GROUP BY chapter ORDER BY start_page", (book_id,)))

SYSTEM_PROMPT = (
    "You are a precise, friendly book-reading companion for students. "
    "Use provided excerpts and Study Atlas entries first. When you infer, say so. "
    "Always cite as (Book <id>, p.<page> — <chapter/atlas>). "
    "For multi-book questions, compare and contrast clearly and align themes directly."
)

def create_thread(name:str, book_ids:list[int]) -> int:
    con = get_db()
    with con:
        cur = con.execute("INSERT INTO threads(name,created_at) VALUES(?,?)", (name, now_iso())); tid = cur.lastrowid
        con.executemany("INSERT INTO thread_books(thread_id,book_id) VALUES(?,?)", [(tid, b) for b in book_ids])
        con.execute("INSERT INTO messages(thread_id, role, content, created_at) VALUES(?,?,?,?)",
                    (tid, "system", SYSTEM_PROMPT, now_iso()))
    return int(tid)

def get_thread(tid:int):
    con = get_db()
    t = con.execute("SELECT * FROM threads WHERE id=?", (tid,)).fetchone()
    books = list(con.execute("SELECT b.* FROM books b JOIN thread_books tb ON b.id=tb.book_id WHERE tb.thread_id=? ORDER BY b.title",(tid,)))
    msgs = list(con.execute("SELECT role,content,created_at FROM messages WHERE thread_id=? ORDER BY id",(tid,)))
    return {"thread": t, "books": books, "messages": msgs}

def append_message(tid:int, role:str, content:str):
    con = get_db()
    with con:
        con.execute("INSERT INTO messages(thread_id,role,content,created_at) VALUES(?,?,?,?)",
                    (tid, role, content, now_iso()))

def list_threads():
    return list(get_db().execute("SELECT id,name,created_at FROM threads ORDER BY id DESC"))

def _run_fts(con: sqlite3.Connection, sql_bm25: str, sql_simple: str, params: list):
    try:
        return list(con.execute(sql_bm25, params))
    except sqlite3.OperationalError:
        return list(con.execute(sql_simple, params))

def search_pages(query:str, book_ids:list[int], k:int=TOPK_DEFAULT, chapter_filters:Optional[Dict[int,List[str]]]=None):
    con = get_db()
    if not book_ids:
        return []
    chapter_filters = chapter_filters or {}
    placeholders = ",".join(["?"] * len(book_ids))

    fts = _fts_query_from_text(query)
    # If no usable terms, fall back to simple selection (no MATCH)
    if not fts:
        sql = (
            f"SELECT p.id,p.book_id,p.page_no,p.chapter,p.text, 1 AS score "
            f"FROM book_pages p WHERE p.book_id IN ({placeholders}) "
        )
        params: list = [*book_ids]
        # Optionally apply chapter filters
        clauses = []
        for bid, pats in chapter_filters.items():
            if not pats: continue
            ors = " OR ".join(["(p.book_id=? AND p.chapter LIKE ?)"] * len(pats))
            clauses.append(f"({ors})")
            for pat in pats:
                params.extend([bid, f"%{pat}%"])
        if clauses:
            sql += " AND (" + " OR ".join(clauses) + ")"
        sql += " ORDER BY p.page_no LIMIT ?"
        params.append(k)
        return list(con.execute(sql, params))

    base = (
        f"SELECT p.id,p.book_id,p.page_no,p.chapter, "
        f"snippet(pages_fts,0,'[',']','...',12) AS snippet, p.text, bm25(pages_fts) AS score "
        f"FROM pages_fts JOIN book_pages p ON p.id = pages_fts.rowid "
        f"WHERE pages_fts MATCH ? AND p.book_id IN ({placeholders})"
    )
    base_simple = base.replace("bm25(pages_fts) AS score", "1 AS score")
    params: list = [fts, *book_ids]
    clauses = []
    for bid, pats in chapter_filters.items():
        if not pats: continue
        ors = " OR ".join(["(p.book_id=? AND p.chapter LIKE ?)"] * len(pats))
        clauses.append(f"({ors})")
        for pat in pats: params.extend([bid, f"%{pat}%"])
    where = base + (" AND (" + " OR ".join(clauses) + ")" if clauses else "")
    where_simple = base_simple + (" AND (" + " OR ".join(clauses) + ")" if clauses else "")
    sql_bm25 = where + " ORDER BY score LIMIT ?"
    sql_simple = where_simple + " LIMIT ?"
    params2 = [*params, k]
    return _run_fts(con, sql_bm25, sql_simple, params2)

def search_annotations(query:str, book_ids:list[int], categories:Optional[List[str]]=None, k:int=20):
    con = get_db()
    if not book_ids:
        return []
    placeholders = ",".join(["?"] * len(book_ids))

    fts = _fts_query_from_text(query)
    categories = categories or []

    if not fts:
        # Fallback without MATCH
        sql = (
            f"SELECT a.id,a.book_id,a.page_no,a.category,a.label,a.details, a.details AS snippet, 1 AS score "
            f"FROM annotations a WHERE a.book_id IN ({placeholders})"
        )
        params: list = [*book_ids]
        if categories:
            cats = ",".join(["?"] * len(categories))
            sql += f" AND a.category IN ({cats})"
            params.extend(categories)
        sql += " ORDER BY a.id DESC LIMIT ?"
        params.append(k)
        return list(con.execute(sql, params))

    base = (
        f"SELECT a.id,a.book_id,a.page_no,a.category,a.label,a.details, "
        f"snippet(annotations_fts,0,'[',']','...',10) AS snippet, bm25(annotations_fts) AS score "
        f"FROM annotations_fts JOIN annotations a ON a.id = annotations_fts.rowid "
        f"WHERE annotations_fts MATCH ? AND a.book_id IN ({placeholders})"
    )
    base_simple = base.replace("bm25(annotations_fts) AS score", "1 AS score")

    params: list = [fts, *book_ids]
    if categories:
        cats = ",".join(["?"] * len(categories))
        base += f" AND a.category IN ({cats})"
        base_simple += f" AND a.category IN ({cats})"
        params.extend(categories)
    sql_bm25 = base + " ORDER BY score LIMIT ?"
    sql_simple = base_simple + " LIMIT ?"
    params.append(k)
    return _run_fts(con, sql_bm25, sql_simple, params)

def build_context_chunks(rows: List[sqlite3.Row], max_chars:int):
    ctx_parts, cites, used = [], [], 0
    for r in rows:
        chunk = r["text"].strip()
        chap = f" • {r['chapter']}" if r["chapter"] else ""
        hdr = f"\\n\\n---\\n[Book {r['book_id']} • Page {r['page_no']}{chap}]\\n"
        piece = hdr + chunk
        if used + len(piece) > max_chars: break
        ctx_parts.append(piece); cites.append((r["book_id"], r["page_no"], r["chapter"])); used += len(piece)
    return "".join(ctx_parts), cites

def build_atlas_context(annos: List[sqlite3.Row], max_chars:int):
    parts, cites, used = [], [], 0
    for a in annos:
        hdr = f"\\n\\n---\\n[Atlas • {a['category']} • {a['label']} • Book {a['book_id']}, p.{a['page_no']}]\\n"
        body = (a["details"] or a["snippet"] or "").strip()
        piece = hdr + body
        if used + len(piece) > max_chars: break
        parts.append(piece); cites.append((a["book_id"], a["page_no"], a["category"], a["label"])); used += len(piece)
    return "".join(parts), cites

BOOK_CHAPTER_PAT = re.compile(r"chapter\\s+(\\d+|[ivxlcdm]+)\\s+of\\s+([^,.;]+)", re.I)
def extract_chapter_directives(user_text:str, available_books:list[sqlite3.Row]):
    directives = {}
    matches = BOOK_CHAPTER_PAT.findall(user_text or "")
    def find_book_id(name:str):
        name = name.strip().lower()
        for b in available_books:
            if name in (_row_get(b, 'title', '') or "").lower(): return int(b["id"])
        return None
    for chap_label, book_name in matches:
        bid = find_book_id(book_name)
        if bid is None: continue
        token = str(chap_label).strip()
        directives.setdefault(bid, []).append(token)
    return directives

def call_openai(model:str, msgs:list[dict]) -> str:
    prefs = get_model_prefs()
    if prefs.get("test_mode"):
        return "🔧 Test mode: Model calls are disabled. This is a simulated answer based on your context and question."
    if not _OPENAI_OK:
        return "⚠️ OpenAI SDK not available."
    key = get_api_key()
    base_url = prefs.get("base_url") or None
    if not key and (not base_url or not base_url.startswith("http://localhost")):
        return "⚠️ API key is not set. Go to Options → Providers to add one, or configure a local provider (e.g., Ollama) with a localhost base URL."
    try:
        client = OpenAI(api_key=key, base_url=base_url)
        use_model = model or prefs.get("model")
        r = client.chat.completions.create(model=use_model, messages=msgs, temperature=0.2)
        return r.choices[0].message.content or ""
    except Exception as e:
        return f"⚠️ Provider error: {e}"

def build_history_digest(book_ids:list[int], max_items:int=20) -> str:
    con = get_db()
    q = """
    SELECT m.role, m.content, m.created_at FROM messages m
    JOIN thread_books tb ON tb.thread_id=m.thread_id
    WHERE tb.book_id IN ({}) ORDER BY m.id DESC LIMIT ?
    """.format(",".join(["?"]*len(book_ids)))
    rows = list(con.execute(q, (*book_ids, max_items)))
    rows.reverse()
    parts = []
    for r in rows:
        role = "Q" if r["role"]=="user" else ("A" if r["role"]=="assistant" else "SYS")
        parts.append(f"{role}: {r['content'][:300]}")
    return "\\n".join(parts)

# ---------------- UI ----------------
st.set_page_config(page_title=f"Book Companion+ ({VERSION_DISPLAY})", page_icon="📚", layout="wide")
st.title(f"📚 Book Companion+ — {VERSION_DISPLAY}")

# Log app version event
try:
    con_init = get_db()
    con_init.execute("INSERT INTO events(ts,event,data_json) VALUES(?,?,?)",
                     (datetime.utcnow().replace(microsecond=0).isoformat()+"Z",
                      "app_version",
                      json.dumps({"version": VERSION, "codename": CODENAME})))
    con_init.commit()
except Exception:
    pass

tabs = st.tabs(["Library","Discussions","Guided Compare","Antagonist","Atlas","Characters","Timeline","Export/Import","Options"])

with tabs[0]:
    st.header("Library")
    files = st.file_uploader("Upload EPUB / TXT / MOBI", type=["epub","txt","mobi"], accept_multiple_files=True)
    if files:
        for f in files:
            try:
                bid = add_book(f.name, "."+f.name.split(".")[-1].lower(), f.read())
                st.success(f"Added '{f.name}' as Book ID {bid}")
            except Exception as e:
                st.error(f"Failed {f.name}: {e}")
    _normalize_book_titles()
    books = list_books()
if books:
    st.subheader("Start a new discussion with selected books")
    ids = [b["id"] for b in books]
    if "lib_books_select_v417" in st.session_state and not set(st.session_state["lib_books_select_v417"]).issubset(set(ids)):
        st.session_state["lib_books_select_v417"] = []
    select_all = st.checkbox("Select entire library", key="lib_select_all_v427_main")
    default_selection = ids if (select_all or len(ids)==1) else []
    selected = st.multiselect(
        "Choose books",
        options=ids,
        default=default_selection,
        key="lib_books_select_v427_main",
        format_func=lambda i: _book_title(i),
    )
    thread_name = st.text_input("Discussion name", value=f"Study Session {datetime.now().strftime('%Y-%m-%d %H:%M')}", key="lib_discussion_name_v427_main")
    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("Start discussion", key="lib_start_discussion_v427_main"):
            if selected:
                tid = create_thread(thread_name, selected)
                st.session_state["last_created_tid"] = tid
                st.success(f"Discussion #{tid} created with {len(selected)} book(s). Go to Discussions tab.")
            else:
                st.warning("Select at least one book.")
    with c2:
        if st.button("Clean up invalid books", key="lib_cleanup_v427"):
            stats = cleanup_invalid_books()
            st.success(f"Cleanup done. Removed books: {stats['books']}, orphaned pages: {stats['orphans']}. Refresh to see changes.")
    st.markdown("---")

    st.subheader("Your books")
    for b in books:
        title = (_row_get(b, 'title', 'Untitled') or 'Untitled').strip()
        meta = f" — {_row_get(b, 'author', 'Unknown')}"
        st.caption(f"ID {b['id']} • {title}{meta}")
    else:
        st.info("No books yet. Upload to begin.")

with tabs[1]:
    st.header("Discussions")

# --- New discussion creation inside this tab ---
all_books = list_books()
with st.expander("➕ Start a new discussion", expanded=(len(all_books) > 0 and len(list_threads()) == 0)):
    if not all_books:
        st.info("Upload a book in the Library tab first.")
    else:
        ids_all = [b["id"] for b in all_books]
        sel_all = st.checkbox("Select entire library", value=(len(ids_all) == 1), key="chat_select_all_v427_main")
        sel_ids = (
            ids_all
            if (sel_all or len(ids_all) == 1)
            else st.multiselect(
                "Choose books for this discussion",
                options=ids_all,
                format_func=lambda i: _book_title(i),
                key="chat_books_select_v427_main"
            )
        )
        default_name = f"Study Session {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        discussion_name = st.text_input("Discussion name", value=default_name, key="chat_discussion_name_v427_main")
        if st.button("Create discussion", key="create_discussion_in_chat_v427_main"):
            if sel_ids:
                new_tid = create_thread(discussion_name, sel_ids)
                st.session_state['last_created_tid'] = new_tid
                st.success(f"Discussion #{new_tid} created. It should be selected below.")
            else:
                st.warning("Select at least one book.")


    # --- New Thread creation inside this tab ---
    all_books = list_books()
    with st.expander("➕ Start a new thread", expanded=(len(all_books) > 0 and len(list_threads()) == 0)):
        if not all_books:
            st.info("Upload a book in the Library tab first.")
        else:
            ids_all = [b["id"] for b in all_books]
            sel_all = st.checkbox("Select entire library", value=(len(ids_all) == 1), key="chat_select_all_v427_inline")
            sel_ids = ids_all if sel_all else st.multiselect(
                "Choose books for this thread",
                options=ids_all,
                format_func=lambda i: _book_title(i),
            )
            default_name = f"Study Session {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            thread_name_new = st.text_input("Thread name", value=default_name, key="new_thread_name_in_chat_v427")
            if st.button("Create thread", key="create_thread_in_chat_v427"):
                if sel_ids:
                    new_tid = create_thread(thread_name_new, sel_ids)
                    st.success(f"Thread #{new_tid} created. Select it below to start chatting.")
                else:
                    st.warning("Select at least one book.")

    # Existing threads
    ths = list_threads()
    if not ths:
        st.info("No threads yet. Use **➕ Start a new thread** above to begin.")
        tid = None
    else:
        tid = st.selectbox(
            "Open thread",
            options=[t["id"] for t in ths],
            format_func=lambda i: _book_title(i),
        )

    if tid:
        state = get_thread(int(tid))
        tinfo, tbooks = state["thread"], state["books"]
        st.subheader(tinfo["name"])
        cols = st.columns([3,2])
        with cols[1]:
            st.markdown("### Retrieval")
            topk = st.slider("Pages to consider", 4, 30, TOPK_DEFAULT, 1, key="chat_topk_v427")
            model = st.text_input("OpenAI model", value=MODEL_DEFAULT, key="chat_model_v427")
            query_hint = st.text_input("Optional targeted search", key="chat_query_hint_v427")
            manual_filters = {}
            with st.expander("Chapter filters"):
                for b in tbooks:
                    chs = list_chapters(int(b["id"]))
                    pretty = [c["chapter"] for c in chs]
                    sel = st.multiselect(f"{_row_get(b, 'title', '')} chapters", options=pretty, default=[], key=f"chap_{b['id']}")
                    if sel:
                        manual_filters[int(b["id"])] = sel
            st.session_state["manual_filters"] = manual_filters
            auto_attach = st.toggle("Auto‑attach context (excerpts + Atlas + summary + past Q/A digest)", value=True, key="chat_auto_attach_v427")

        with cols[0]:
            st.markdown("### Discussion")
            for m in state["messages"]:
                if m["role"] == "system":
                    continue
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])
            prompt = st.chat_input("Ask about themes, characters, patterns… or compare books.")
            if prompt:
                append_message(tinfo["id"], "user", prompt)
                with st.chat_message("user"):
                    st.markdown(prompt)

                book_ids = [b["id"] for b in tbooks]
                q = (query_hint.strip() or prompt)
                auto_dirs = extract_chapter_directives(prompt, tbooks)
                manual = st.session_state.get("manual_filters", {}) or {}
                merged_filters = {}
                for bid in {**{b: None for b in auto_dirs.keys()}, **{b: None for b in manual.keys()}}.keys():
                    merged_filters[bid] = list(set((auto_dirs.get(bid, []) + manual.get(bid, []))))

                page_rows = search_pages(q, book_ids, k=topk, chapter_filters=merged_filters)
                ctx_pages, cites_pages = build_context_chunks(page_rows, max_chars=CHUNK_CHAR_BUDGET*3)

                anno_rows = search_annotations(q or '*', book_ids, k=20)
                ctx_atlas, cites_atlas = build_atlas_context(anno_rows, max_chars=CHUNK_CHAR_BUDGET*2)

                con = get_db()
                row = con.execute(
                    "SELECT summary FROM study_summaries WHERE book_id IN (" + (",".join(["?"] * len(book_ids))) + ") ORDER BY id DESC LIMIT 1",
                    book_ids
                ).fetchone()
                ctx_summary = ("\n\n---\n[Recent Study Summary]\n" + row["summary"]) if row else ""

                history_digest = build_history_digest(book_ids, max_items=30)

                user_content = f"Question: {prompt}\n\n"
                if auto_attach:
                    if ctx_pages:
                        user_content += "Relevant excerpts:" + ctx_pages
                    if ctx_atlas:
                        user_content += "\n\nAtlas entries:" + ctx_atlas
                    user_content += (ctx_summary or "")
                    if history_digest:
                        user_content += "\n\n---\n[Past Q/A Digest]\n" + history_digest
                    if not ctx_pages and not ctx_atlas and not ctx_summary and not history_digest:
                        user_content += "\n(⚠️ No indexed excerpts found; answer generally and say so.)"

                messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}]
                try:
                    with st.spinner("Thinking…"):
                        answer = call_openai(model, messages)
                except Exception as e:
                    answer = f"⚠️ OpenAI error: {e}"

                with st.chat_message("assistant"):
                    st.markdown(answer)
                    if cites_pages or cites_atlas:
                        st.caption("Sources:")
                        for (bid, pno, ch) in sorted(set(cites_pages)):
                            title = _book_title(bid)
                            chs = f" — {ch}" if ch else ""
                            st.caption(f"• {title}{chs} — p.{pno} (id:{bid})")
                        for (bid, pno, cat, label) in cites_atlas:
                            title = _book_title(bid)
                            st.caption(f"• Atlas: {cat} — {label} — {title} p.{pno}")
                append_message(tinfo["id"], "assistant", answer)

with tabs[2]:
    st.header("Guided Compare")
    books = list_books()
    if len(books) < 2:
        st.info("Upload at least two books to compare.")
    else:
        format_func=lambda i: _book_title(i),
        format_func=lambda i: _book_title(i),
        if 'b1' in locals() and 'b2' in locals() and b1 and b2:
            chs1 = list_chapters(int(b1))
            chs2 = list_chapters(int(b2))
        else:
            chs1, chs2 = [], []  # fallback to empty lists if not selected
        c1 = st.selectbox("Chapter (A)", [c["chapter"] for c in chs1], key="c1")
        c2 = st.selectbox("Chapter (B)", [c["chapter"] for c in chs2], key="c2")
        model = st.text_input("OpenAI model", value=MODEL_DEFAULT, key="cmp_model")
        limit_pages = st.slider("Pages per chapter", 1, 12, 5, 1)
        if st.button("Compare chapters"):
            con = get_db()
            def pull(book_id:int, chapter:str, limit_pages:int=5):
                return list(con.execute("SELECT page_no,chapter,text FROM book_pages WHERE book_id=? AND chapter LIKE ? ORDER BY page_no LIMIT ?",
                                        (book_id, f"%{chapter}%", limit_pages)))
            ex1 = pull(int(b1), c1, limit_pages); ex2 = pull(int(b2), c2, limit_pages)
            st.write("**Excerpts (A)**")
            for e in ex1:
                st.caption(f"p.{e['page_no']} — {e['chapter']}"); st.text(e['text'][:2000])
            st.write("**Excerpts (B)**")
            for e in ex2:
                st.caption(f"p.{e['page_no']} — {e['chapter']}"); st.text(e['text'][:2000])
            comp_prompt = (f"Compare theme, style, and language.\nA: id {b1}, chapter {c1}. B: id {b2}, chapter {c2}.\nUse quotes; cite (Book <id>, p.<page> — <chapter>).")
            ctx = []
            for e in ex1:
                ctx.append(f"\n[Book {b1} • p.{e['page_no']} • {e['chapter']}]\n{e['text']}")
            for e in ex2:
                ctx.append(f"\n[Book {b2} • p.{e['page_no']} • {e['chapter']}]\n{e['text']}")
            messages = [{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":comp_prompt+"\n\nRelevant excerpts:"+"\n".join(ctx)}]
            try:
                with st.spinner("Analyzing…"): ans = call_openai(model, messages)
            except Exception as e:
                ans = f"⚠️ OpenAI error: {e}"
            st.markdown(ans)

with tabs[3]:
    st.header("🗣️ The Antagonist (Counter‑Debate)")
    books = list_books()
if books:
    st.subheader("Start a new discussion with selected books")
    ids = [b["id"] for b in books]
    if "lib_books_select_v421_b" in st.session_state and not set(st.session_state["lib_books_select_v421_b"]).issubset(set(ids)):
        st.session_state["lib_books_select_v421_b"] = []
    select_all = st.checkbox("Select entire library", key="lib_select_all_v427_footer")
    default_selection = ids if (select_all or len(ids)==1) else []
    selected = st.multiselect(
        "Choose books",
        options=ids,
        default=default_selection,
        key="lib_books_select_v427_footer",
        format_func=lambda i: _book_title(i),
    )
    thread_name = st.text_input("Discussion name", value=f"Study Session {datetime.now().strftime('%Y-%m-%d %H:%M')}", key="lib_discussion_name_v427_main")
    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("Start discussion", key="lib_start_discussion_v427_main"):
            if selected:
                tid = create_thread(thread_name, selected)
                st.session_state["last_created_tid"] = tid
                st.success(f"Discussion #{tid} created with {len(selected)} book(s). Go to Discussions tab.")
            else:
                st.warning("Select at least one book.")
    with c2:
        if st.button("Clean up invalid books", key="lib_cleanup_v427"):
            stats = cleanup_invalid_books()
            st.success(f"Cleanup done. Removed books: {stats['books']}, orphaned pages: {stats['orphans']}. Refresh to see changes.")
    st.markdown("---")
    st.subheader("OpenAI API Key")
    current = get_api_key()
    status = "✅ Key set" if current else "⚠️ Not set"
    st.caption(f"Status: {status}")
    api_key_input = st.text_input("Enter/Update API Key", value=current or "", type="password", key="opt_api_key_v427")
    cols = st.columns([1,1])
    with cols[0]:
        if st.button("Save key", key="opt_save_key_v427"):
            if api_key_input.strip():
                import os as _os
                _prefs_set("OPENAI_API_KEY", api_key_input.strip())
                st.session_state["OPENAI_API_KEY"] = api_key_input.strip()
                _os.environ["OPENAI_API_KEY"] = api_key_input.strip()
                st.success("API key saved for this app. (Stored in local DB user_prefs.)")
            else:
                st.warning("Please paste a valid key before saving.")
    with cols[1]:
        if st.button("Clear key", key="opt_clear_key_v427"):
            _prefs_set("OPENAI_API_KEY", "")
            if "OPENAI_API_KEY" in st.session_state:
                del st.session_state["OPENAI_API_KEY"]
            import os as _os
            if "OPENAI_API_KEY" in _os.environ:
                del _os.environ["OPENAI_API_KEY"]
            st.success("API key cleared for this app.")
    st.markdown("---")
    st.caption("Keys are stored locally in the app's SQLite database (user_prefs table).")

    # Footer badge
    st.markdown(f"<div style='text-align:right; opacity:0.6; font-size:12px;'>Book Companion+ {VERSION_DISPLAY}</div>", unsafe_allow_html=True)