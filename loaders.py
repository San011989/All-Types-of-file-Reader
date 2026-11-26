"""File loaders: turn PDF / CSV / Excel / TXT uploads into LangChain Documents.

Tabular files (CSV/Excel) are NOT chunked like prose. Splitting a table every
1000 characters cuts rows in half and separates values from their column
names, which is why answers were unreliable. Instead:

  * every row is rewritten as  "Column: value | Column: value ..."
  * rows are grouped into chunks of whole rows, each chunk repeating the
    file / sheet / column header so it is understandable on its own
  * one extra "schema" chunk per sheet describes columns, row count and
    simple statistics (used for summary-type questions)
"""
import os
from io import BytesIO

import pandas as pd
from langchain_core.documents import Document

TABULAR_EXTS = {".csv", ".xlsx", ".xls"}
TEXT_EXTS = {".txt", ".md"}
SUPPORTED_EXTS = ["pdf", "csv", "xlsx", "xls", "txt", "md"]

# Max characters of row text per chunk (whole rows only).
ROW_CHUNK_CHARS = 1500


def get_ext(filename: str) -> str:
    return os.path.splitext(filename)[1].lower()


# ---------------------------------------------------------------------
# Tabular
# ---------------------------------------------------------------------

def _read_csv(data: bytes) -> pd.DataFrame:
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            # sep=None + python engine -> auto-detects , ; \t |
            return pd.read_csv(
                BytesIO(data), sep=None, engine="python",
                encoding=encoding, dtype=str, keep_default_na=False, header=None,
            )
        except Exception as error:  # try next encoding / fail at the end
            last_error = error
    raise ValueError(f"Could not read CSV file: {last_error}")


def _read_excel(data: bytes, ext: str) -> dict:
    engine = "openpyxl" if ext == ".xlsx" else "xlrd"
    return pd.read_excel(
        BytesIO(data), sheet_name=None, engine=engine,
        dtype=str, keep_default_na=False, header=None,
    )


def _norm(value) -> str:
    """Single-line, stripped text ('Name\n(123)' -> 'Name (123)')."""
    text = " ".join(str(value).split())
    return "" if text in ("nan", "NaT", "None") else text


def _promote_header(raw: pd.DataFrame):
    """Find the real header row (spreadsheets often have blank rows or a title
    above it). Returns (dataframe_with_header, title_text)."""
    raw = raw.apply(lambda col: col.map(_norm))
    raw = raw.loc[~(raw == "").all(axis=1)].reset_index(drop=True)
    raw = raw.loc[:, ~(raw == "").all(axis=0)]
    if raw.empty:
        return raw, ""

    counts = (raw != "").sum(axis=1)
    window = counts.iloc[:30]
    needed = max(2, int(0.6 * window.max() + 0.999))
    header_idx = next((i for i, c in window.items() if c >= needed), 0)

    title = " ".join(
        " ".join(v for v in raw.iloc[i].tolist() if v) for i in range(header_idx)
    ).strip()

    names, seen = [], {}
    for i, v in enumerate(raw.iloc[header_idx].tolist()):
        name = v or f"Column {i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name} ({seen[name]})"
        else:
            seen[name] = 1
        names.append(name)

    body = raw.iloc[header_idx + 1:].copy()
    body.columns = names
    return body.reset_index(drop=True), title


def _schema_text(df: pd.DataFrame, source: str, sheet: str, title: str = "") -> str:
    first = df.columns[0]
    is_total = df[first].str.lower().str.match(r"^(grand\s+)?total\b")
    stats_df = df.loc[~is_total]
    lines = [
        f"TABLE OVERVIEW AND SCHEMA",
        f"File: {source}",
        f"Sheet: {sheet}",
        *([f"Sheet title: {title}"] if title else []),
        f"Total rows: {len(df)}",
        f"Total columns: {len(df.columns)}",
        f"Column names: {', '.join(df.columns)}",
        "",
        "Column details:",
    ]
    for col in df.columns:
        series = stats_df[col]
        non_empty = series[series != ""]
        if len(non_empty) == 0:
            lines.append(f"- {col}: no values (all empty)")
            continue
        numeric = pd.to_numeric(non_empty.str.replace(",", "", regex=False),
                                errors="coerce")
        if len(non_empty) and numeric.notna().mean() > 0.9:
            lines.append(
                f"- {col} (numeric): min={numeric.min():g}, max={numeric.max():g}, "
                f"mean={numeric.mean():.4g}, sum={numeric.sum():.6g}, "
                f"missing={len(series) - len(non_empty)}"
            )
        else:
            uniques = non_empty.unique()
            if len(uniques) <= 15:
                shown = ", ".join(map(str, uniques))
                lines.append(f"- {col} (text, {len(uniques)} unique values): {shown}")
            else:
                sample = ", ".join(map(str, uniques[:5]))
                lines.append(
                    f"- {col} (text, {len(uniques)} unique values), e.g. {sample}"
                )
    return "\n".join(lines)


def _dataframe_to_documents(df: pd.DataFrame, source: str, sheet: str) -> list:
    df, title = _promote_header(df)
    if df.empty:
        return []

    docs = []
    header = (
        f"File: {source} | Sheet: {sheet} | "
        + (f"Title: {title} | " if title else "")
        + f"Columns: {', '.join(df.columns)}"
    )

    # 1) schema / overview chunk (also returned for "summarize" questions)
    docs.append(Document(
        page_content=_schema_text(df, source, sheet, title),
        metadata={"source": source, "sheet": sheet, "type": "overview"},
    ))

    # 2) row chunks made of whole rows
    buffer, buffer_len, start_row = [], 0, 1

    def flush(end_row):
        nonlocal buffer, buffer_len
        if buffer:
            docs.append(Document(
                page_content=f"{header} | Rows {start_row}-{end_row}\n" + "\n".join(buffer),
                metadata={
                    "source": source, "sheet": sheet, "type": "table_rows",
                    "row_start": start_row, "row_end": end_row,
                },
            ))
        buffer, buffer_len = [], 0

    for idx, row in enumerate(df.itertuples(index=False), start=1):
        parts = [f"{col}: {val}" for col, val in zip(df.columns, row) if val != ""]
        line = f"Row {idx} | " + " | ".join(parts)
        if buffer and buffer_len + len(line) > ROW_CHUNK_CHARS:
            flush(idx - 1)
            start_row = idx
        buffer.append(line)
        buffer_len += len(line) + 1
    flush(len(df))
    return docs


def _read_frames(data: bytes, filename: str) -> dict:
    ext = get_ext(filename)
    if ext == ".csv":
        return {"CSV": _read_csv(data)}
    try:
        return _read_excel(data, ext)
    except ImportError as error:
        raise RuntimeError(
            "Reading this Excel format needs 'openpyxl' (.xlsx) or "
            "'xlrd' (.xls). Add them to requirements.txt."
        ) from error


def load_tabular(data: bytes, filename: str) -> list:
    docs = []
    for sheet, df in _read_frames(data, filename).items():
        docs.extend(_dataframe_to_documents(df, filename, str(sheet)))
    return docs


def load_tables(data: bytes, filename: str) -> dict:
    """{sheet_name: cleaned DataFrame with the real header row}"""
    out = {}
    for sheet, df in _read_frames(data, filename).items():
        clean, _title = _promote_header(df)
        if not clean.empty:
            out[str(sheet)] = clean
    return out


# ---------------------------------------------------------------------
# Stored copies of the cleaned tables (used to show the data directly)
# ---------------------------------------------------------------------

def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def save_tables(directory: str, source: str, frames: dict):
    os.makedirs(directory, exist_ok=True)
    delete_tables(directory, source)
    for sheet, df in frames.items():
        path = os.path.join(directory, f"{_safe(source)}__{_safe(sheet)}.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")


def delete_tables(directory: str, source: str):
    if not os.path.isdir(directory):
        return
    prefix = f"{_safe(source)}__"
    for fname in os.listdir(directory):
        if fname.startswith(prefix):
            os.remove(os.path.join(directory, fname))


def tables_to_markdown(directory: str, limit: int = 50) -> str:
    """Markdown preview (first `limit` rows) of every stored table."""
    if not os.path.isdir(directory):
        return ""
    parts = []
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(directory, fname), dtype=str,
                         keep_default_na=False, encoding="utf-8-sig")
        source, _, sheet = fname[:-4].partition("__")
        shown = df.head(limit)
        esc = lambda v: str(v).replace("|", "\\|")
        lines = [
            f"**{source}** — sheet **{sheet}** "
            f"(showing {len(shown)} of {len(df)} rows)",
            "",
            "| " + " | ".join(esc(c) for c in df.columns) + " |",
            "| " + " | ".join("---" for _ in df.columns) + " |",
        ]
        for row in shown.itertuples(index=False):
            lines.append("| " + " | ".join(esc(v) for v in row) + " |")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------

def load_text(data: bytes, filename: str) -> list:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if not text.strip():
        return []
    return [Document(page_content=text, metadata={"source": filename})]
