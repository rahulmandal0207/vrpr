"""Rebuild src/vrpr/data/rto_codes.csv from Wikipedia's list of RTO districts in India.

Run it from the repository root when the list changes:

    python scripts/build_rto_codes.py

The table is CC BY-SA 4.0, credited in src/vrpr/data/CREDITS.md. VRPR itself never goes online
to describe a plate: it reads only the CSV this script writes.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

PAGE = "List of Regional Transport Office districts in India"
API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "vrpr-rto-table-builder/1.0 (https://github.com/rahulmandal0207/vrpr)"
OUTPUT = Path(__file__).resolve().parent.parent / "src" / "vrpr" / "data" / "rto_codes.csv"

_DASH = "-" + chr(0x2013) + chr(0x2014)  # hyphen, en dash and em dash: Wikipedia uses all three
_HEADING = re.compile(rf"^\s*([A-Z]{{2}})\s*[{_DASH}]\s*(.+?)\s*$")
_CODE = re.compile(rf"\b([A-Z]{{2}})\s*[{_DASH}]?\s*0*(\d{{1,3}})\b")
_RANGE = re.compile(
    rf"\b([A-Z]{{2}})\s*[{_DASH}]?\s*0*(\d{{1,3}})\s*(?:to|[{_DASH}])"
    rf"\s*(?:[A-Z]{{2}}\s*[{_DASH}]?\s*)?0*(\d{{1,3}})\b"
)
_REFERENCE = re.compile(r"\[\d+\]|\[citation needed\]|\[[a-z]\]")
_SPACE = re.compile(r"\s+")
_ASIDE = re.compile(r"\s*\((?:previously|formerly|earlier)[^)]*\)", re.IGNORECASE)
_LABEL = re.compile(r"^(?:jurisdiction|office|area)\s*:\s*", re.IGNORECASE)
_SKIPPED = ("script", "style", "sup")


class _Tables(HTMLParser):
    """Collect each wikitable's cells, with the heading of the section it is in."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[tuple[str, list[list[tuple[str, int, int]]]]] = []
        self._heading: list[str] | None = None
        self._section = ""
        self._depth = 0  # nesting of wikitables
        self._row: list[tuple[str, int, int]] | None = None
        self._cell: list[str] | None = None
        self._span = (1, 1)
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag in _SKIPPED:
            self._skip += 1
        elif tag == "h2":
            self._heading = []
        elif tag == "table" and "wikitable" in (values.get("class") or ""):
            self._depth += 1
            if self._depth == 1:
                self.tables.append((self._section, []))
        elif self._depth == 1 and tag == "tr":
            self._row = []
        elif self._depth == 1 and tag in ("td", "th") and self._row is not None:
            self._cell = []
            self._span = (_number(values.get("rowspan")), _number(values.get("colspan")))
        elif tag == "br" and self._cell is not None:
            self._cell.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIPPED:
            self._skip = max(0, self._skip - 1)
        elif tag == "h2" and self._heading is not None:
            self._section = "".join(self._heading).strip()
            self._heading = None
        elif tag == "table" and self._depth:
            self._depth -= 1
        elif self._depth == 1 and tag in ("td", "th") and self._cell is not None:
            if self._row is None:
                return
            self._row.append(("".join(self._cell), *self._span))
            self._cell = None
        elif self._depth == 1 and tag == "tr" and self._row is not None:
            self.tables[-1][1].append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self._heading is not None:
            self._heading.append(data)
        elif self._cell is not None:
            self._cell.append(data)


def _number(value: str | None) -> int:
    try:
        return max(1, int(value or 1))
    except ValueError:
        return 1


def _clean(text: str, first_line: bool = False) -> str:
    text = _REFERENCE.sub("", text)
    if first_line:
        text = next((line for line in text.splitlines() if line.strip()), "")
    text = _LABEL.sub("", _ASIDE.sub("", _SPACE.sub(" ", text).strip()))
    if first_line:
        text = text.split(". ")[0]  # the area itself, without the explanation after it
    return text.rstrip(":.").strip()


def _grid(rows: list[list[tuple[str, int, int]]]) -> list[list[str]]:
    """Expand rowspan and colspan, so every row has a value in every column."""
    pending: dict[int, tuple[str, int]] = {}
    grid: list[list[str]] = []
    for row in rows:
        cells = iter(row)
        out: list[str] = []
        column = 0
        while True:
            if column in pending:
                text, left = pending.pop(column)
                out.append(text)
                if left > 1:
                    pending[column] = (text, left - 1)
                column += 1
                continue
            cell = next(cells, None)
            if cell is None:
                break
            text, rowspan, colspan = cell
            for _ in range(colspan):
                out.append(text)
                if rowspan > 1:
                    pending[column] = (text, rowspan - 1)
                column += 1
        while column in pending:  # spans that continue past this row's last cell
            text, left = pending.pop(column)
            out.append(text)
            if left > 1:
                pending[column] = (text, left - 1)
            column += 1
        grid.append(out)
    return grid


def _codes(text: str, state: str) -> list[int]:
    numbers: list[int] = []
    for match in _RANGE.finditer(text):
        if match.group(1) == state and int(match.group(2)) < int(match.group(3)) <= 999:
            numbers.extend(range(int(match.group(2)), int(match.group(3)) + 1))
    for match in _CODE.finditer(text):
        if match.group(1) == state:
            numbers.append(int(match.group(2)))
    return sorted(set(numbers))


def _column(header: list[str], *names: str) -> int | None:
    for index, title in enumerate(header):
        if any(name in title.lower() for name in names):
            return index
    return None


def fetch_html() -> str:
    query = urllib.parse.urlencode(
        {"action": "parse", "page": PAGE, "prop": "text", "format": "json", "formatversion": "2"}
    )
    # A fixed https URL, so the scheme checks these rules ask for do not apply.
    request = urllib.request.Request(f"{API}?{query}", headers={"User-Agent": USER_AGENT})  # noqa: S310
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return str(json.load(response)["parse"]["text"])


def build_rows(html: str) -> list[dict[str, str]]:
    parser = _Tables()
    parser.feed(html)
    found: dict[tuple[str, int], dict[str, str]] = {}
    for section, rows in parser.tables:
        heading = _HEADING.match(section)
        if heading is None or not rows:
            continue
        state, state_name = heading.group(1), _clean(heading.group(2))
        grid = _grid(rows)
        header = [_clean(cell).lower() for cell in grid[0]]
        code_at = _column(header, "code")
        office_at = _column(header, "office", "location")
        area_at = _column(header, "jurisdiction", "area", "district")
        note_at = _column(header, "annotation", "remark", "note")
        if code_at is None:
            continue
        for cells in grid[1:]:

            def cell(index: int | None, first_line: bool = False, row: list[str] = cells) -> str:
                return (
                    _clean(row[index], first_line) if index is not None and index < len(row) else ""
                )

            office, area, note = cell(office_at), cell(area_at, first_line=True), cell(note_at)
            for number in _codes(cells[code_at] if code_at < len(cells) else "", state):
                key = (state, number)
                if key in found:
                    # A merged code cell repeats for each district it covers. Keep every area.
                    existing = found[key]
                    if area and area not in existing["jurisdiction"].split("; "):
                        existing["jurisdiction"] = "; ".join(
                            part for part in (existing["jurisdiction"], area) if part
                        )
                    continue
                found[key] = {
                    "state_code": state,
                    "rto": str(number),
                    "state": state_name,
                    "office": office,
                    "jurisdiction": area,
                    "note": note,
                }
    return [found[key] for key in sorted(found)]


def main() -> int:
    rows = build_rows(fetch_html())
    if len(rows) < 1000:
        print(f"Only {len(rows)} codes found; the page layout may have changed.", file=sys.stderr)
        return 1
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    states = len({row["state_code"] for row in rows})
    print(f"Wrote {len(rows)} RTO codes for {states} states and territories to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
