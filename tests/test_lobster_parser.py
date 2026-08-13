"""Parser tests: LOBSTER CSV -> canonical schema frames.

Inputs are tiny hand-written CSVs in a tmp dir whose expected frames are known
by hand. No raw LOBSTER data is touched (the raw files are never committed and
these tests must run on CI without them).
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.lobster import read_message_frame, read_orderbook_frame
from src.schema import EVENT_COLUMNS, EVENT_DTYPES, book_columns, validate_event_frame

# --------------------------------------------------------------------------- #
# Message parser
# --------------------------------------------------------------------------- #


def test_read_message_frame_maps_columns_and_dtypes(tmp_path):
    # Two raw LOBSTER message rows: time,type,order_id,size,price,direction.
    csv = tmp_path / "msg.csv"
    csv.write_text(
        "34200.005742728,1,16114545,100,275200,-1\n34200.006461694,4,16114695,50,275500,1\n"
    )
    df = read_message_frame(csv)

    assert list(df.columns) == list(EVENT_COLUMNS)
    for col, dtype in EVENT_DTYPES.items():
        assert str(df[col].dtype) == dtype
    # Positional map: time->time_s, type->event_type, ... direction->direction.
    assert df["time_s"].iloc[0] == pytest.approx(34200.005742728)
    assert df["event_type"].iloc[0] == 1
    assert df["order_id"].iloc[1] == 16114695
    assert df["size"].iloc[1] == 50
    assert df["price"].iloc[0] == 275200
    assert list(df["direction"]) == [-1, 1]


def test_read_message_frame_output_passes_schema_validation(tmp_path):
    csv = tmp_path / "msg.csv"
    csv.write_text("1.0,1,1,10,100,1\n2.0,4,1,5,100,1\n3.0,3,2,7,101,-1\n")
    df = read_message_frame(csv)
    validate_event_frame(df)  # should not raise


def test_read_message_frame_ignores_trailing_columns(tmp_path):
    # LOBSTER message files carry exactly six columns; guard usecols by adding a
    # stray seventh field and confirming only the first six are consumed.
    csv = tmp_path / "msg.csv"
    csv.write_text("1.0,1,1,10,100,1,999\n")
    df = read_message_frame(csv)
    assert list(df.columns) == list(EVENT_COLUMNS)
    assert df.shape == (1, 6)


# --------------------------------------------------------------------------- #
# Orderbook parser
# --------------------------------------------------------------------------- #


def test_read_orderbook_frame_layout_and_dtype(tmp_path):
    # One level-2 book row: ask_px_1,ask_sz_1,bid_px_1,bid_sz_1,ask_px_2,...
    csv = tmp_path / "ob.csv"
    csv.write_text("110,20,100,10,111,5,99,7\n")
    df = read_orderbook_frame(csv, levels=2)

    assert list(df.columns) == book_columns(2)
    assert str(df.to_numpy().dtype) == "int64"
    assert df["ask_px_1"].iloc[0] == 110
    assert df["bid_sz_1"].iloc[0] == 10
    assert df["ask_px_2"].iloc[0] == 111
    assert df["bid_sz_2"].iloc[0] == 7


def test_read_orderbook_frame_uses_only_requested_levels(tmp_path):
    # A level-10-wide row read at levels=1 keeps only the first four columns.
    row = ",".join(str(v) for v in range(40))
    csv = tmp_path / "ob.csv"
    csv.write_text(row + "\n")
    df = read_orderbook_frame(csv, levels=1)
    assert list(df.columns) == book_columns(1)
    assert df.shape == (1, 4)
    assert list(df.iloc[0]) == [0, 1, 2, 3]


def test_read_orderbook_frame_rejects_bad_levels(tmp_path):
    csv = tmp_path / "ob.csv"
    csv.write_text("110,20,100,10\n")
    with pytest.raises(ValueError, match="levels must be"):
        read_orderbook_frame(csv, levels=0)


def test_message_and_orderbook_are_row_aligned(tmp_path):
    # The two files describe the same session row-for-row; a correct read keeps
    # equal row counts (the differential relies on this alignment).
    msg = tmp_path / "msg.csv"
    msg.write_text("1.0,1,1,10,100,1\n2.0,1,2,20,110,-1\n")
    ob = tmp_path / "ob.csv"
    ob.write_text("110,0,100,10\n110,20,100,10\n")
    events = read_message_frame(msg)
    book = read_orderbook_frame(ob, levels=1)
    assert len(events) == len(book)


def test_read_message_frame_accepts_str_and_path(tmp_path):
    csv = tmp_path / "msg.csv"
    csv.write_text("1.0,1,1,10,100,1\n")
    from_path = read_message_frame(csv)
    from_str = read_message_frame(str(csv))
    pd.testing.assert_frame_equal(from_path, from_str)
