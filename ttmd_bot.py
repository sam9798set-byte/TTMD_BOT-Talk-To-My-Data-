"""
TTMD Bot: AI Business Data Copilot (backend + command-line version)

Question -> embedding -> similar examples from ChromaDB -> few-shot prompt
-> Groq writes SQL -> SQL is validated -> MySQL runs it -> Groq explains result.

Files:
    ttmd_bot.py       this file (application logic, used by the CLI and by app.py)
    ttmd_examples.py  the Question -> SQL example library
    app.py            the Streamlit web interface

Run (command line):
    python ttmd_bot.py              normal start
    python ttmd_bot.py --rebuild    delete and rebuild the ChromaDB example store
    python ttmd_bot.py --selftest   test the SQL validator only (no DB, no API key)

Run (web interface):
    python -m streamlit run app.py

Credentials:
    Paste your Groq key and MySQL password into GROQ_KEY and DB_PASSWORD in the
    SETTINGS section below. Keep this file private and do not share it.
"""

import os

# Turn off ChromaDB's anonymous usage reporting (must be set before importing chroma).
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import hashlib
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import quote_plus

import groq
import sqlglot
from groq import Groq
from langchain_chroma import Chroma
from langchain_community.utilities import SQLDatabase
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from sqlalchemy import create_engine, event, inspect, text
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError

from ttmd_examples import EXAMPLES


# =====================================================================
# SETTINGS: edit these (paste your key and password below)
# =====================================================================
GROQ_KEY = "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"          # <- paste your Groq API key between the quotes
MODEL = "openai/gpt-oss-120b"

DB_USER = "root"
DB_PASSWORD = "R@hulisGr8"      # <- paste your MySQL password between the quotes
DB_HOST = "localhost"
DB_PORT = 3306
DB_NAME = "ttmd"
TABLE = "modeldata"


def _clean_setting(value) -> str:
    """A pasted value can end with a stray comma (Python then builds a tuple) or have
    spaces/new lines around it. Always return a clean text value."""
    if isinstance(value, (tuple, list)):
        value = value[0] if value else ""
    return str(value).strip()


GROQ_KEY = _clean_setting(GROQ_KEY)
DB_PASSWORD = _clean_setting(DB_PASSWORD)

# Business definitions the AI must follow (kept from your existing code).
# Edit if "unreliable" means something else in your data.
BUSINESS_RULES = """
- An "active" device is one where Recency_Category = '1. Active (< 24 hrs)'.
- A "stale" device is one where Recency_Category = '4. Stale (> 30 Days)'.
- An "unreliable" device is the same as a stale device.
- Percentage of devices in a group = devices in the group divided by total devices in
  that scope, multiplied by 100.
- To compare categories (for example active vs stale) by device type, group by
  DeviceTypeName and count the devices in each category.
- "Most stale devices" means the highest COUNT of stale devices (rank by the count),
  while "highest percentage of stale devices" means rank by the percentage.
"""

# Embeddings + vector store
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = "chroma_db"
CHROMA_COLLECTION = "ttmd_sql_examples"
NUM_EXAMPLES = 3            # how many similar examples go into each prompt

# Limits
MAX_SQL_ATTEMPTS = 3        # first try + up to 2 corrections (MySQL error or a wrong UNANSWERABLE)
DEBUG_MODEL_OUTPUT = False  # set True to print the model's raw reply in the terminal
MAX_ROWS = 50               # rows fetched from MySQL
MAX_RESULT_CHARS = 3000     # result text sent to Groq for the final answer
PREVIEW_ROWS = 5            # rows shown in the terminal (CLI only)
QUERY_TIMEOUT_MS = 30000    # MySQL stops SELECTs running longer than this
MAX_SQL_LENGTH = 5000

# Columns with these words in their name are never shown to the AI as sample values.
SENSITIVE_WORDS = ["phone", "mobile", "address", "email", "contact", "imei", "sim", "iccid"]


# =====================================================================
# Errors (so generation, validation and execution problems stay separate)
# =====================================================================
class TTMDError(Exception):
    label = "Error"


class StartupError(TTMDError):
    label = "Startup error"


class LLMError(TTMDError):
    label = "Groq error"


class SQLGenerationError(TTMDError):
    label = "SQL generation error"


class SQLValidationError(TTMDError):
    label = "SQL blocked by validation"


class SQLExecutionError(TTMDError):
    label = "SQL execution error"


def redact(message: Any) -> str:
    """Remove secrets from any text before printing or showing it."""
    msg = str(message)
    for secret in (GROQ_KEY, DB_PASSWORD, quote_plus(DB_PASSWORD)):
        if secret and not secret.startswith("your-"):
            msg = msg.replace(secret, "***")
    return msg


def _db_error_text(e: Exception) -> str:
    """Short, readable database error (for example: 1054 Unknown column 'x' ...)."""
    orig = getattr(e, "orig", None)
    if orig is not None and getattr(orig, "args", None):
        return redact(" ".join(str(a) for a in orig.args))[:300]
    return redact(e)[:300]


def _friendly_connect_error(e: Exception) -> str:
    raw = _db_error_text(e)
    low = raw.lower()
    if "1045" in raw or "access denied" in low:
        return "MySQL rejected the username or password. Check DB_USER and DB_PASSWORD at the top of ttmd_bot.py."
    if "1049" in raw or "unknown database" in low:
        return f"The database '{DB_NAME}' does not exist on the MySQL server."
    if "2003" in raw or "can't connect" in low or "connection refused" in low:
        return f"Cannot reach MySQL at {DB_HOST}:{DB_PORT}. Is the MySQL service running?"
    return f"Could not connect to MySQL: {raw}"


# =====================================================================
# Structured result (what the CLI prints and what Streamlit displays)
# =====================================================================
@dataclass
class QueryResult:
    question: str
    status: str = "error"                 # "success" | "unanswerable" | "error"
    answer: str = ""
    sql: Optional[str] = None             # last SQL the model generated
    executed: bool = False                # True only if MySQL actually ran the SQL
    columns: list = field(default_factory=list)
    rows: list = field(default_factory=list)      # list of plain tuples
    truncated: bool = False               # True if more than MAX_ROWS rows matched
    examples: list = field(default_factory=list)  # questions of the retrieved examples
    attempts: int = 0
    error_type: Optional[str] = None
    error_message: Optional[str] = None


# =====================================================================
# 1. MySQL connection
# =====================================================================
def connect_mysql():
    """Connect to MySQL with a read-only, time-limited session."""
    uri = (
        f"mysql+pymysql://{DB_USER}:{quote_plus(DB_PASSWORD)}"
        f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    engine = create_engine(uri, pool_pre_ping=True)

    @event.listens_for(engine, "connect")
    def _harden_session(dbapi_connection, _record):
        # Extra safety layer: MySQL itself refuses writes in this session.
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("SET SESSION TRANSACTION READ ONLY")
            cursor.execute(f"SET SESSION MAX_EXECUTION_TIME={int(QUERY_TIMEOUT_MS)}")
        except Exception:
            pass  # verified right after connecting, see below
        finally:
            cursor.close()

    db = SQLDatabase(engine, sample_rows_in_table_info=0)  # no sample rows sent to the AI

    with engine.connect() as conn:
        read_only = conn.execute(text("SELECT @@session.transaction_read_only")).scalar()
    print("Read-only MySQL session:", "ON" if read_only else "OFF (validator is still active)")
    return engine, db


def get_value_hints(engine, table=TABLE, max_values=15, max_len=60) -> str:
    """List the real values of columns that only have a few distinct values,
    so the AI uses your exact labels (for example '4. Stale (> 30 Days)')."""
    hints = []
    try:
        cols = [c["name"] for c in inspect(engine).get_columns(table)]
        with engine.connect() as conn:
            for col in cols:
                if any(w in col.lower() for w in SENSITIVE_WORDS):
                    continue
                n = conn.execute(text(f"SELECT COUNT(DISTINCT `{col}`) FROM `{table}`")).scalar()
                if n and n <= max_values:
                    rows = conn.execute(text(f"SELECT DISTINCT `{col}` FROM `{table}`")).fetchall()
                    vals = [str(r[0]) for r in rows]
                    if all(len(v) <= max_len for v in vals):
                        hints.append(f"{col}: {vals}")
    except Exception as e:
        print("Note: could not build value hints:", redact(e)[:200])
    return "\nPossible values for category columns:\n" + "\n".join(hints)


def load_schema(engine, db) -> str:
    tables = db.get_usable_table_names()
    if TABLE not in tables:
        raise StartupError(f"Table '{TABLE}' was not found in database '{DB_NAME}'. Found: {tables}")
    info = db.get_table_info([TABLE])
    if not info or not info.strip():
        raise StartupError(f"Could not read the structure of table '{TABLE}'.")
    return info + get_value_hints(engine)


def show_schema(engine):
    print(f"\nTable {TABLE}:")
    for col in inspect(engine).get_columns(TABLE):
        print(f"  {col['name']}  ({col['type']})")


# =====================================================================
# 2. SQL validation (parser based; never relies on the AI behaving)
# =====================================================================
def _node_types(names):
    return tuple(getattr(exp, n) for n in names if hasattr(exp, n))


FORBIDDEN_NODES = _node_types([
    "Insert", "Update", "Delete", "Drop", "Create", "Alter", "Merge", "TruncateTable",
    "Command", "Set", "Use", "Into", "Lock", "Transaction", "Commit", "Rollback",
    "Parameter", "SessionParameter", "Placeholder",
])
ALLOWED_ROOTS = _node_types(["Select", "Union", "SetOperation"])
DANGEROUS_FUNCTIONS = {
    "sleep", "benchmark", "load_file", "get_lock", "release_lock", "release_all_locks",
    "is_free_lock", "is_used_lock", "sys_exec", "sys_eval", "master_pos_wait", "source_pos_wait",
}
_LITERAL = re.compile(r"'(?:[^'\\]|\\.|'')*'|\"(?:[^\"\\]|\\.|\"\")*\"|`[^`]*`")


def _raw_text_guard(sql: str) -> None:
    """Reject comments and statement separators outside quoted text."""
    bare = _LITERAL.sub("''", sql)
    for token, reason in (
        (";", "multiple statements"),
        ("--", "SQL comments"),
        ("#", "SQL comments"),
        ("/*", "SQL comments"),
        ("*/", "SQL comments"),
    ):
        if token in bare:
            raise SQLValidationError(f"Rejected: {reason} are not allowed.")


def validate_sql(sql: str, allowed_tables) -> None:
    """Raise SQLValidationError unless `sql` is ONE read-only SELECT that only
    touches known tables. Anything that cannot be confidently verified is rejected."""
    if not sql or not sql.strip():
        raise SQLValidationError("Rejected: the query is empty.")
    if len(sql) > MAX_SQL_LENGTH:
        raise SQLValidationError("Rejected: the query is too long.")

    _raw_text_guard(sql)

    try:
        tokens = sqlglot.tokenize(sql, read="mysql")
        statements = [s for s in sqlglot.parse(sql, read="mysql") if s is not None]
    except (ParseError, TokenError):
        raise SQLValidationError("Rejected: the query could not be parsed, so it cannot be verified as safe.")
    except Exception:
        raise SQLValidationError("Rejected: the query could not be checked.")

    if any(tok.comments for tok in tokens):
        raise SQLValidationError("Rejected: SQL comments are not allowed.")
    if len(statements) != 1:
        raise SQLValidationError("Rejected: exactly one statement is allowed.")

    root = statements[0]
    if not isinstance(root, ALLOWED_ROOTS):
        raise SQLValidationError("Rejected: only read-only SELECT queries are allowed.")

    for node in root.walk():
        if isinstance(node, FORBIDDEN_NODES):
            raise SQLValidationError(f"Rejected: disallowed SQL element ({type(node).__name__}).")
        if isinstance(node, exp.Func):
            name = node.name if isinstance(node, exp.Anonymous) else node.sql_name()
            if str(name).lower() in DANGEROUS_FUNCTIONS:
                raise SQLValidationError(f"Rejected: function {name} is not allowed.")

    allowed = {t.lower() for t in allowed_tables}
    cte_names = {str(cte.alias).lower() for cte in root.find_all(exp.CTE)}
    for table in root.find_all(exp.Table):
        name = str(table.name).lower()
        if not name:
            raise SQLValidationError("Rejected: unsupported table expression.")
        if table.catalog or (table.db and table.db.lower() != DB_NAME.lower()):
            raise SQLValidationError("Rejected: other databases cannot be queried.")
        if name not in allowed and name not in cte_names:
            raise SQLValidationError(f"Rejected: unknown table '{table.name}'.")


def run_validator_selftest() -> None:
    allowed = {"modeldata"}
    should_pass = [
        "SELECT COUNT(*) FROM modeldata",
        "SELECT DeviceTypeName, COUNT(*) AS n FROM modeldata WHERE DeviceTypeName LIKE '%SIMHA%' GROUP BY DeviceTypeName",
        "WITH t AS (SELECT DeviceTypeName FROM modeldata) SELECT COUNT(*) FROM t",
        "SELECT * FROM modeldata WHERE note = 'a;b -- c'",
    ]
    should_fail = [
        "DELETE FROM modeldata",
        "UPDATE modeldata SET Recency_Category = 'x'",
        "DROP TABLE modeldata",
        "SELECT * FROM modeldata; DROP TABLE modeldata",
        "WITH t AS (SELECT 1) DELETE FROM modeldata",
        "SELECT * FROM modeldata -- hidden",
        "SELECT * FROM modeldata /* hidden */",
        "SELECT * FROM modeldata # hidden",
        "SELECT * FROM other_table",
        "SELECT * FROM mysql.user",
        "SELECT SLEEP(10)",
        "SELECT * FROM modeldata FOR UPDATE",
        "SELECT * FROM modeldata INTO OUTFILE '/tmp/x'",
        "SHOW TABLES",
        "",
    ]
    bad = 0
    print("Validator self-test\n")
    for sql in should_pass:
        try:
            validate_sql(sql, allowed)
            print("  PASS (allowed) :", sql[:70])
        except SQLValidationError as e:
            bad += 1
            print("  FAIL (should be allowed):", sql[:70], "->", e)
    for sql in should_fail:
        try:
            validate_sql(sql, allowed)
            bad += 1
            print("  FAIL (should be blocked):", sql[:70])
        except SQLValidationError:
            print("  PASS (blocked) :", sql[:70] or "<empty>")
    print(f"\n{'All checks passed.' if bad == 0 else str(bad) + ' check(s) failed.'}")


# =====================================================================
# 3. Example library, embeddings and ChromaDB
# =====================================================================
def example_id(example: dict) -> str:
    """Same question + same SQL always gives the same id, so nothing is stored twice."""
    raw = example["question"] + "\n" + example["query"]
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def check_examples(engine, allowed_tables) -> list:
    """Keep only examples that pass the validator AND that MySQL can plan (EXPLAIN)."""
    valid = []
    for ex in EXAMPLES:
        question = str(ex.get("question", "")).strip()
        query = str(ex.get("query", "")).strip().rstrip(";").strip()
        try:
            if not question or not query:
                raise ValueError("empty question or query")
            if "{" in query or "}" in query:
                raise ValueError("curly braces are not allowed in example queries")
            validate_sql(query, allowed_tables)
            with engine.connect() as conn:
                conn.execute(text("EXPLAIN " + query))
        except Exception as e:
            print(f"  Skipping example '{question[:55]}': {redact(e)[:160]}")
            continue
        valid.append({
            "question": question,
            "query": query,
            "category": str(ex.get("category", "general")),
        })
    print(f"Examples validated against your database: {len(valid)} of {len(EXAMPLES)}")
    return valid


def init_embeddings():
    """Hugging Face model that turns text into vectors (downloaded once, then cached)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def _open_store(embeddings):
    return Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


def init_vector_store(embeddings, examples: list, rebuild: bool = False):
    """Create or reuse the persistent ChromaDB store and sync it with the example list."""
    store = _open_store(embeddings)
    if rebuild:
        store.delete_collection()
        store = _open_store(embeddings)

    wanted = {example_id(ex): ex for ex in examples}
    existing = set(store.get()["ids"])
    to_remove = list(existing - set(wanted))
    to_add = [i for i in wanted if i not in existing]

    if to_remove:
        store.delete(ids=to_remove)
    if to_add:
        store.add_texts(
            texts=[wanted[i]["question"] for i in to_add],   # this text is embedded
            metadatas=[
                {
                    "question": wanted[i]["question"],
                    "query": wanted[i]["query"],
                    "category": wanted[i]["category"],
                }
                for i in to_add
            ],
            ids=to_add,
        )
    print(f"Vector store '{CHROMA_DIR}': {len(wanted)} examples (added {len(to_add)}, removed {len(to_remove)})")
    return store


def build_selector(store):
    """Finds the NUM_EXAMPLES stored questions most similar in meaning to the new question."""
    return SemanticSimilarityExampleSelector(
        vectorstore=store,
        k=NUM_EXAMPLES,
        example_keys=["question", "query"],   # what is returned for each match
        input_keys=["question"],              # what is embedded when searching
    )


def retrieve_examples(selector, question: str) -> list:
    """Retrieve similar examples for display. Returns [] if retrieval is unavailable."""
    if selector is None:
        return []
    try:
        return selector.select_examples({"question": question})
    except Exception as e:
        print("  Warning: example retrieval failed, using schema-only prompt:", redact(e)[:200])
        return []


# =====================================================================
# 4. Prompt construction (LangChain)
# =====================================================================
EXAMPLE_PROMPT = PromptTemplate.from_template("Question: {question}\nSQL Query: {query}")

PREFIX_BASE = """You are an expert MySQL analyst. Convert the user's question into ONE SQL query.

Instructions:
- The database dialect is MySQL.
- Write exactly ONE read-only SELECT statement (a WITH ... SELECT is allowed).
- Use only the tables and columns shown in the schema below. Never invent names or values.
- Use the exact category values listed in the schema when filtering.
- Use the business definitions below whenever the question uses those terms.
- Always try to answer. Counts, percentages, rankings, comparisons, filters and groupings
  of this table can all be written as SQL. Choose the most reasonable interpretation
  using the schema and the definitions.
- If the query returns a list of rows, add LIMIT 50.
- Output only the SQL: no markdown, no comments, no explanation, no trailing semicolon.
- Output the single word UNANSWERABLE only if the question has nothing to do with the
  data in this table (for example weather or general knowledge).

Database schema (the only source of truth for table and column names):
{schema}

Business definitions:
{rules}
"""

PREFIX_WITH_EXAMPLES = PREFIX_BASE + """
Below are example questions with correct SQL queries for similar questions.
Use them only as guidance for style and structure. They may not match the current
question exactly, so always follow the schema above."""

PREFIX_NO_EXAMPLES = PREFIX_BASE

SUFFIX = "{retry_note}Question: {question}\nSQL Query:"

PROMPT_VARIABLES = ["schema", "rules", "question", "retry_note"]

ANSWER_PROMPT = PromptTemplate.from_template(
    """You are a business data assistant. Answer the user's question using ONLY the SQL result below.

Question: {question}

Result columns: {columns}
Result rows ({row_count} row(s) shown{truncation_note}):
{rows}

Rules:
- Answer in one to three plain English sentences.
- Do not mention SQL, tables or columns.
- Do not invent numbers that are not in the result.
- If the result is cut off, say the figures are based on the rows shown."""
)


@dataclass
class BotContext:
    engine: Any
    schema: str
    allowed_tables: set
    client: Any
    selector: Optional[Any] = None
    dynamic_prompt: Optional[Any] = field(default=None)
    fallback_prompt: Optional[Any] = field(default=None)
    example_count: int = 0                                  # validated examples available
    startup_warnings: list = field(default_factory=list)    # shown by the web interface


def make_prompt_templates(selector):
    dynamic = None
    if selector is not None:
        dynamic = FewShotPromptTemplate(
            example_selector=selector,          # examples are chosen per question
            example_prompt=EXAMPLE_PROMPT,
            prefix=PREFIX_WITH_EXAMPLES,
            suffix=SUFFIX,
            input_variables=PROMPT_VARIABLES,
            example_separator="\n\n",
        )
    fallback = FewShotPromptTemplate(
        examples=[],                            # schema-only prompt, no examples
        example_prompt=EXAMPLE_PROMPT,
        prefix=PREFIX_NO_EXAMPLES,
        suffix=SUFFIX,
        input_variables=PROMPT_VARIABLES,
        example_separator="\n\n",
    )
    return dynamic, fallback


def build_prompt(ctx: BotContext, question: str, retry_note: str = "", use_examples: bool = True) -> str:
    values = dict(schema=ctx.schema, rules=BUSINESS_RULES, question=question, retry_note=retry_note)
    if use_examples and ctx.dynamic_prompt is not None:
        try:
            return ctx.dynamic_prompt.format(**values)
        except Exception as e:
            print("  Warning: few-shot prompt failed, using schema-only prompt:", redact(e)[:200])
    return ctx.fallback_prompt.format(**values)


# =====================================================================
# 5. Groq: SQL generation and answer generation
# =====================================================================
RETRYABLE = tuple(
    getattr(groq, name)
    for name in ("RateLimitError", "InternalServerError", "APIConnectionError", "APITimeoutError")
    if hasattr(groq, name)
)


def _groq_status_message(status: int, err: Exception) -> str:
    known = {
        401: "Groq rejected the API key. Check GROQ_KEY at the top of ttmd_bot.py.",
        403: "Groq denied access for this key or model.",
        404: f"Groq could not find the model '{MODEL}' for your account. Check MODEL.",
        413: "The request was too large for the model.",
    }
    return known.get(status, f"Groq returned status {status}: {redact(err)[:200]}")


def ask_llm(client, prompt: str, retries: int = 4) -> str:
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
        except RETRYABLE as e:
            wait = 2 ** attempt
            print(f"  Groq is busy ({type(e).__name__}). Retrying in {wait}s...")
            time.sleep(wait)
            continue
        except groq.APIStatusError as e:
            raise LLMError(_groq_status_message(e.status_code, e))
        except Exception as e:
            raise LLMError(redact(e)[:300])

        content = response.choices[0].message.content if response.choices else None
        if not content or not content.strip():
            raise LLMError("Groq returned an empty response.")
        return content.strip()
    raise LLMError("Groq is still busy after several tries. Please try again in a few minutes.")


UNANSWERABLE_MARK = "UNANSWERABLE"


def extract_sql(raw: str) -> str:
    """Pull the SQL out of the model's reply. Returns UNANSWERABLE_MARK if the model
    declined, or "" if no SQL could be found."""
    body = raw.strip()
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", body, flags=re.S | re.I)
    if fenced:
        body = fenced.group(1).strip()
    body = re.sub(r"^\s*SQL Query:\s*", "", body, flags=re.I).strip()

    if body.upper().startswith(UNANSWERABLE_MARK):
        return UNANSWERABLE_MARK

    if not re.match(r"(?is)^\s*(select|with)\b", body):
        # The model added some text before the query: keep only the query part.
        found = re.search(r"(?is)\b(select\b.*|with\s+\w+\s+as\s*\(.*)", body)
        body = found.group(1).strip() if found else ""

    if body.endswith(";"):
        body = body[:-1].rstrip()
    return body


def generate_sql(ctx: BotContext, prompt: str) -> str:
    """Ask Groq for the SQL. Returns the SQL text, or UNANSWERABLE_MARK if the model declined."""
    raw = ask_llm(ctx.client, prompt)
    if DEBUG_MODEL_OUTPUT:
        print("RAW GROQ RESPONSE:", repr(raw))

    sql = extract_sql(raw)
    if not sql:
        raise SQLGenerationError(f"The model did not return a SQL query. Its reply started with: {raw[:150]!r}")
    return sql


def generate_answer(ctx: BotContext, question: str, columns, rows, truncated: bool) -> str:
    if not rows:
        return "The query ran successfully but returned no rows, so no matching data was found."
    rows_text = str([tuple(r) for r in rows])
    cut = len(rows_text) > MAX_RESULT_CHARS
    prompt = ANSWER_PROMPT.format(
        question=question,
        columns=", ".join(columns),
        row_count=len(rows),
        truncation_note=" - more rows exist" if (truncated or cut) else "",
        rows=rows_text[:MAX_RESULT_CHARS],
    )
    return ask_llm(ctx.client, prompt)


# =====================================================================
# 6. SQL execution
# =====================================================================
def execute_sql(engine, sql: str):
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            columns = list(result.keys())
            rows = result.fetchmany(MAX_ROWS + 1)
    except Exception as e:
        raise SQLExecutionError(_db_error_text(e))
    truncated = len(rows) > MAX_ROWS
    return columns, rows[:MAX_ROWS], truncated


# =====================================================================
# 7. The pipeline: returns a QueryResult (used by the CLI and by app.py)
# =====================================================================
UNANSWERABLE_TEXT = "This question can't be answered from the columns available in the data."
UNANSWERABLE_RETRY_NOTE = (
    "Your previous answer was UNANSWERABLE, but this question can almost certainly be answered "
    "from the table above using filtering, grouping, counting, percentages or comparisons. "
    "Use the exact column names and category values shown in the schema and the business "
    "definitions, and write the SQL query now. Answer UNANSWERABLE again only if the question "
    "has nothing to do with this table.\n\n"
)


def answer_question(ctx: BotContext, question: str, log=None) -> QueryResult:
    """Run the whole pipeline for one question. Never raises: problems are returned
    inside the QueryResult. `log` is an optional function (the CLI passes print)."""
    log = log or (lambda *_args, **_kwargs: None)
    result = QueryResult(question=question)

    try:
        examples = retrieve_examples(ctx.selector, question)
        result.examples = [ex["question"] for ex in examples]
        if examples:
            log("Retrieved similar examples:")
            for ex in examples:
                log(f"  - {ex['question']}")
        else:
            log("No similar examples available, using the schema-only prompt.")

        retry_note = ""
        columns, rows, truncated = [], [], False

        for attempt in range(1, MAX_SQL_ATTEMPTS + 1):
            result.attempts = attempt
            prompt = build_prompt(ctx, question, retry_note, use_examples=bool(examples))
            sql = generate_sql(ctx, prompt)

            if sql == UNANSWERABLE_MARK:
                if attempt < MAX_SQL_ATTEMPTS:
                    log("  The model answered UNANSWERABLE. Asking it to try again...")
                    retry_note = UNANSWERABLE_RETRY_NOTE
                    continue
                result.status = "unanswerable"
                result.answer = UNANSWERABLE_TEXT
                return result

            result.sql = sql
            log(f"\nGenerated SQL (attempt {attempt}):\n{sql}")

            validate_sql(sql, ctx.allowed_tables)   # raises SQLValidationError; nothing is executed

            try:
                columns, rows, truncated = execute_sql(ctx.engine, sql)
                result.executed = True
                break
            except SQLExecutionError as e:
                if attempt == MAX_SQL_ATTEMPTS:
                    raise
                log(f"  MySQL rejected the query, asking the model to correct it: {e}")
                retry_note = (
                    "Your previous attempt failed.\n"
                    f"Previous SQL: {sql}\n"
                    f"MySQL error: {e}\n"
                    "Write a corrected query.\n\n"
                )

        result.columns = list(columns)
        result.rows = [tuple(r) for r in rows]
        result.truncated = truncated

        result.answer = generate_answer(ctx, question, result.columns, result.rows, truncated)
        result.status = "success"

    except TTMDError as e:
        result.status = "error"
        result.error_type = e.label
        result.error_message = str(e)
    except Exception as e:
        result.status = "error"
        result.error_type = "Unexpected error"
        result.error_message = redact(f"{type(e).__name__}: {e}")[:300]

    return result


# =====================================================================
# 8. Command-line chat loop (unchanged behaviour, now built on answer_question)
# =====================================================================
def _print_rows(result: QueryResult) -> None:
    print(f"\nRows returned: {len(result.rows)}{'+' if result.truncated else ''}")
    if result.rows:
        print("Columns:", ", ".join(result.columns))
        for row in result.rows[:PREVIEW_ROWS]:
            print("  ", tuple(row))
        if len(result.rows) > PREVIEW_ROWS:
            print(f"   ... {len(result.rows) - PREVIEW_ROWS} more row(s) not shown")


def handle_question(ctx: BotContext, question: str) -> None:
    print(f"\nQuestion: {question}")
    result = answer_question(ctx, question, log=print)

    if result.status == "unanswerable":
        print(f"\n{result.answer}")
        return
    if result.executed:
        _print_rows(result)
    if result.status == "error":
        print(f"\n[{result.error_type}] {result.error_message}")
        return
    print("\nAnswer:", result.answer)


def chat_loop(ctx: BotContext) -> None:
    print("\nTTMD Bot is ready. Type 'schema' to list columns.")
    while True:
        try:
            question = input("\nAsk a question about your data (or type 'exit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() == "exit":
            break
        if question.lower() == "schema":
            show_schema(ctx.engine)
            continue
        try:
            handle_question(ctx, question)
        except Exception as e:
            print(f"\n[Unexpected error] {redact(e)[:300]}")


# =====================================================================
# 9. Start-up (shared by the CLI and by app.py)
# =====================================================================
def find_settings_problems() -> list:
    problems = []
    if GROQ_KEY.startswith("your-"):
        problems.append("GROQ_KEY is still the placeholder.")
    if DB_PASSWORD.startswith("your-"):
        problems.append("DB_PASSWORD is still the placeholder.")
    return problems


def initialize_backend(rebuild: bool = False) -> BotContext:
    """Connect to MySQL, load the schema, examples, embeddings, ChromaDB and Groq.
    Raises StartupError (never calls sys.exit) so any front end can show the message."""
    problems = find_settings_problems()
    if problems:
        raise StartupError(
            " ".join(problems)
            + " Edit the SETTINGS section at the top of ttmd_bot.py."
        )

    try:
        engine, db = connect_mysql()
        print("Connected successfully! Tables:", db.get_usable_table_names())
        schema = load_schema(engine, db)
    except TTMDError:
        raise
    except Exception as e:
        raise StartupError(_friendly_connect_error(e))

    allowed_tables = {t.lower() for t in db.get_usable_table_names()}
    examples = check_examples(engine, allowed_tables)
    warnings = []
    if len(examples) < len(EXAMPLES):
        warnings.append(
            f"{len(EXAMPLES) - len(examples)} example(s) were skipped because they did not "
            "match your database. See the terminal for details."
        )

    selector = None
    try:
        print(f"Loading embedding model {EMBEDDING_MODEL} (first run downloads it)...")
        embeddings = init_embeddings()
        store = init_vector_store(embeddings, examples, rebuild=rebuild)
        selector = build_selector(store)
    except Exception as e:
        print("Warning: embeddings or ChromaDB could not start. Continuing WITHOUT examples.")
        print("  Reason:", redact(e)[:300])
        warnings.append(
            "Example retrieval is unavailable (embeddings or ChromaDB failed to start), "
            "so answers use the database schema only."
        )

    dynamic, fallback = make_prompt_templates(selector)
    return BotContext(
        engine=engine,
        schema=schema,
        allowed_tables=allowed_tables,
        client=Groq(api_key=GROQ_KEY),
        selector=selector,
        dynamic_prompt=dynamic,
        fallback_prompt=fallback,
        example_count=len(examples),
        startup_warnings=warnings,
    )


def main() -> None:
    if "--selftest" in sys.argv:
        run_validator_selftest()
        return

    try:
        ctx = initialize_backend(rebuild="--rebuild" in sys.argv)
    except TTMDError as e:
        print(f"[{e.label}] {e}")
        sys.exit(1)
    chat_loop(ctx)


if __name__ == "__main__":
    main()