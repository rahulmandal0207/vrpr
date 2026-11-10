"""Describe what a registration number itself says: where it was issued and what kind it is.

Everything here comes from the number and a bundled table of RTO codes, so it works offline and
reveals nothing about the vehicle's owner. VRPR deliberately does not look plates up in Vahan or
third-party registries: see docs/privacy.md.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from functools import cache
from importlib import resources

from vrpr.plates import BHARAT, STANDARD, parse_plate

_STANDARD_PARTS = re.compile(r"([A-Z]{2})([0-9]{1,2})([A-Z]{1,3})([0-9]{4})")
_BHARAT_PARTS = re.compile(r"([0-9]{2})BH([0-9]{4})([A-Z]{1,2})")
# Codes that states have since replaced. The RTO numbers carried over unchanged.
_RENAMED_STATES = {"OR": "OD", "TS": "TG", "UA": "UK", "DN": "DD"}
_STATES_WITHOUT_TABLE = {"AN": "Andaman and Nicobar Islands"}
# Beyond this many districts, an RTO is described by its count rather than by name.
_MAX_AREAS_LISTED = 3


@dataclass(frozen=True, slots=True)
class RtoOffice:
    state_code: str
    rto: int
    state: str
    office: str
    jurisdiction: str
    note: str


@dataclass(frozen=True, slots=True)
class PlateDetails:
    plate: str
    plate_format: str
    state: str = ""
    rto_code: str = ""
    rto_office: str = ""
    rto_area: str = ""
    registered_year: int | None = None
    note: str = ""

    @property
    def summary(self) -> str:
        """One line for people, such as 'Noida RTO (UP-16), Gautam Buddha Nagar district'."""
        if self.plate_format == BHARAT:
            year = f", first registered in {self.registered_year}" if self.registered_year else ""
            return f"Bharat series, valid in every state{year}"
        place = ", ".join(part for part in (self.rto_area, self.state) if part)
        if self.rto_office:
            kind = "" if "RTO" in self.rto_office.upper().split() else " RTO"
            office = f"{self.rto_office}{kind} ({self.rto_code})"
            return f"{office}, {place}" if place else office
        if self.rto_code:
            return f"{self.rto_code}, {place}" if place else self.rto_code
        return place

    def to_fields(self) -> dict[str, str | int | None]:
        """The columns added to plates.csv and plates.jsonl."""
        return {
            "state": self.state,
            "rto_code": self.rto_code,
            "rto_office": self.rto_office,
            "rto_area": self.rto_area,
            "registered_year": self.registered_year,
        }


@cache
def rto_offices() -> dict[tuple[str, int], RtoOffice]:
    """Load the bundled RTO table, keyed by state code and RTO number."""
    table = resources.files("vrpr").joinpath("data").joinpath("rto_codes.csv")
    text = table.read_text(encoding="utf-8")
    return {
        (row["state_code"], int(row["rto"])): RtoOffice(
            row["state_code"],
            int(row["rto"]),
            row["state"],
            row["office"],
            row["jurisdiction"],
            row["note"],
        )
        for row in csv.DictReader(io.StringIO(text, newline=""))
    }


@cache
def state_names() -> dict[str, str]:
    names = {office.state_code: office.state for office in rto_offices().values()}
    names.update(_STATES_WITHOUT_TABLE)
    for old, new in _RENAMED_STATES.items():
        if new in names:
            names[old] = names[new]
    return names


def _area(jurisdiction: str) -> str:
    areas = [area for area in jurisdiction.split("; ") if area]
    if len(areas) > _MAX_AREAS_LISTED:
        return f"{len(areas)} districts"
    return ", ".join(areas)


# Placeholders in the table that name no place: retired codes, and offices given only by type.
_NOT_A_PLACE = frozenset({"defunct", "rta", "rto"})


def _usable(text: str) -> str:
    return "" if text.strip().lower() in _NOT_A_PLACE else text


def describe_plate(plate: str) -> PlateDetails | None:
    """Describe a plate, or return None if it is not a valid Indian registration number."""
    parsed = parse_plate(plate)
    if parsed is None:
        return None
    if parsed.plate_format == BHARAT:
        match = _BHARAT_PARTS.fullmatch(parsed.plate)
        year = 2000 + int(match.group(1)) if match else None
        return PlateDetails(parsed.plate, BHARAT, registered_year=year)

    match = _STANDARD_PARTS.fullmatch(parsed.plate)
    if match is None:
        return PlateDetails(parsed.plate, STANDARD)
    state_code, rto = match.group(1), int(match.group(2))
    rto_code = f"{state_code}-{rto:02d}"
    state = state_names().get(state_code, "")
    office = rto_offices().get((_RENAMED_STATES.get(state_code, state_code), rto))
    note = ""
    if state_code in _RENAMED_STATES:
        note = f"{state_code} is the former code for {_RENAMED_STATES[state_code]}."
    if office is None:
        return PlateDetails(parsed.plate, STANDARD, state, rto_code, note=note)
    office_name, area = _usable(office.office), _usable(_area(office.jurisdiction))
    if not office_name and not area:
        note = " ".join(part for part in (note, office.note or "This RTO code is retired.") if part)
    return PlateDetails(
        parsed.plate,
        STANDARD,
        state=office.state,
        rto_code=rto_code,
        rto_office=office_name,
        rto_area=area,
        note=note,
    )
