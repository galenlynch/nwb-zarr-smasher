"""Missing electrode values must be filled before they reach pynwb."""

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from hdmf_zarr import NWBZarrIO
from pynwb import NWBFile

from nwb_zarr_smasher.nwb_zarr_smasher import _fill_missing


def _nwb_with_group():
    """A minimal NWBFile carrying one electrode group."""
    nwb = NWBFile(
        session_description="test",
        identifier="test",
        session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    device = nwb.create_device("probe")
    group = nwb.create_electrode_group(
        name="46113", description="test", device=device, location="unknown"
    )
    return nwb, group


class FillMissingTest(unittest.TestCase):
    """``_fill_missing`` substitutes only for genuinely missing scalars."""

    def test_none_and_nan_are_filled(self):
        """The two shapes a missing value actually arrives in."""
        self.assertEqual(_fill_missing(None, ""), "")
        self.assertEqual(_fill_missing(np.nan, ""), "")
        self.assertTrue(np.isnan(_fill_missing(None, np.nan)))

    def test_real_values_pass_through(self):
        """Present values are never substituted, including falsy ones."""
        self.assertEqual(_fill_missing("CA3", ""), "CA3")
        self.assertEqual(_fill_missing(0, np.nan), 0)
        self.assertEqual(_fill_missing("", "fill"), "")

    def test_non_scalars_pass_through(self):
        """An array has no scalar truth value; it must not raise."""
        value = np.array([1.0, 2.0])
        np.testing.assert_array_equal(_fill_missing(value, ""), value)


class DocvalStripsNoneTest(unittest.TestCase):
    """Why the fill is needed at all -- pynwb drops ``None`` kwargs."""

    def test_none_omits_the_column(self):
        """A single ``None`` fails, naming a column present on every other row.

        This is the regression: one CCF-background channel among thousands took
        down the whole packaging run.
        """
        nwb, group = _nwb_with_group()
        nwb.add_electrode_column(name="ccf_acronym", description="test")
        nwb.add_electrode(group=group, location="CA3", ccf_acronym="CA3")

        with self.assertRaises(Exception) as caught:
            nwb.add_electrode(group=group, location="root", ccf_acronym=None)
        self.assertIn("ccf_acronym", str(caught.exception))

    def test_filled_value_is_accepted(self):
        """The same row succeeds once the missing value is filled."""
        nwb, group = _nwb_with_group()
        nwb.add_electrode_column(name="ccf_acronym", description="test")
        nwb.add_electrode(group=group, location="CA3", ccf_acronym="CA3")
        nwb.add_electrode(
            group=group,
            location="root",
            ccf_acronym=_fill_missing(None, ""),
        )
        self.assertEqual(list(nwb.electrodes["ccf_acronym"].data), ["CA3", ""])


class ZarrWriteMixedDtypeTest(unittest.TestCase):
    """The failure that actually killed a run: mixed float/str reaching zarr.

    An electrode group with no alignment leaves a block of float NaN in the text
    ccf_* columns. hdmf-zarr infers the dataset dtype from that leading NaN and
    then chokes on the first real acronym, deep inside the write with no mention
    of which column.
    """

    def _write(self, acronyms, path):
        """Build a one-column electrodes table and write it as zarr."""
        nwb, group = _nwb_with_group()
        nwb.add_electrode_column(name="ccf_acronym", description="test")
        for value in acronyms:
            nwb.add_electrode(group=group, location="x", ccf_acronym=value)
        with NWBZarrIO(str(path), "w") as io:
            io.write(nwb)

    def test_nan_then_string_fails(self):
        """Reproduces `could not convert string to float: 'alv'`."""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as caught:
                self._write([np.nan, np.nan, "alv"], Path(tmp) / "bad.nwb")
            self.assertIn("could not convert string to float",
                          str(caught.exception))

    def test_filled_column_writes(self):
        """The same data writes once the NaNs are filled for a text column."""
        raw = [np.nan, np.nan, "alv"]
        filled = [_fill_missing(v, "") for v in raw]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "good.nwb"
            self._write(filled, path)
            with NWBZarrIO(str(path), "r") as io:
                written = list(io.read().electrodes["ccf_acronym"].data)
        self.assertEqual(written, ["", "", "alv"])


class FillValuesByDtypeTest(unittest.TestCase):
    """Fills follow the column dtype so a text column stays text."""

    def test_text_gets_empty_string_numeric_gets_nan(self):
        """Mirrors the ``fill_values`` mapping built at the write boundary."""
        merged = pd.DataFrame(
            {"ccf_acronym": ["CA3", None], "ccf_id": [463.0, np.nan]}
        )
        fills = {
            c: (np.nan if pd.api.types.is_numeric_dtype(merged[c]) else "")
            for c in merged.columns
        }
        self.assertEqual(fills["ccf_acronym"], "")
        self.assertTrue(np.isnan(fills["ccf_id"]))


if __name__ == "__main__":
    unittest.main()
