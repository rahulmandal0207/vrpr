"""Details read from a registration number and the bundled RTO table."""

import pytest

from vrpr.plate_details import describe_plate, rto_offices, state_names
from vrpr.plates import BHARAT, STANDARD


def test_the_bundled_table_covers_the_country() -> None:
    offices = rto_offices()
    assert len(offices) > 1000
    assert len({state for state, _ in offices}) >= 30


def test_a_standard_plate_names_its_rto() -> None:
    details = describe_plate("UP16AU6426")
    assert details is not None
    assert details.plate_format == STANDARD
    assert (details.state, details.rto_code, details.rto_office, details.rto_area) == (
        "Uttar Pradesh",
        "UP-16",
        "Noida",
        "Gautam Buddha Nagar district",
    )
    assert details.summary == "Noida RTO (UP-16), Gautam Buddha Nagar district, Uttar Pradesh"


def test_delhi_one_digit_codes_are_padded() -> None:
    details = describe_plate("DL3CAB1234")
    assert details is not None
    assert details.rto_code == "DL-03"
    assert details.state == "Delhi"


def test_a_bharat_plate_gives_its_year_and_no_state() -> None:
    details = describe_plate("22BH1234AB")
    assert details is not None
    assert details.plate_format == BHARAT
    assert details.registered_year == 2022
    assert details.state == ""
    assert details.summary == "Bharat series, valid in every state, first registered in 2022"


@pytest.mark.parametrize(("old", "new"), [("TS09EA1234", "TG"), ("OR02AB1234", "OD")])
def test_former_state_codes_use_the_new_table(old: str, new: str) -> None:
    details = describe_plate(old)
    assert details is not None
    assert details.rto_code.startswith(old[:2])
    assert details.rto_office
    assert details.note == f"{old[:2]} is the former code for {new}."


def test_a_retired_code_says_so_instead_of_naming_a_place() -> None:
    details = describe_plate("UP05AB1234")
    assert details is not None
    assert details.rto_office == details.rto_area == ""
    assert "Uttarakhand" in details.note


def test_a_code_missing_from_the_table_still_names_the_state() -> None:
    details = describe_plate("AP28CC4284")
    assert details is not None
    assert (details.state, details.rto_code, details.rto_office) == ("Andhra Pradesh", "AP-28", "")
    assert details.summary == "AP-28, Andhra Pradesh"


def test_a_statewide_code_lists_districts_by_count() -> None:
    details = describe_plate("AP39AB1234")
    assert details is not None
    assert details.rto_area.endswith("districts")
    assert details.rto_office == ""  # the table gives only "RTA", which names no place


def test_every_state_the_parser_accepts_has_a_name() -> None:
    from vrpr.plates import STATE_CODES

    assert set(state_names()) >= STATE_CODES


@pytest.mark.parametrize("text", ["XX12AB1234", "HELLO", ""])
def test_text_that_is_not_a_plate_has_no_details(text: str) -> None:
    assert describe_plate(text) is None


def test_details_fill_the_record_columns() -> None:
    details = describe_plate("KA41AA0033")
    assert details is not None
    fields = details.to_fields()
    assert fields["state"] == "Karnataka"
    assert fields["rto_code"] == "KA-41"
    assert fields["registered_year"] is None
