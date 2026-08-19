"""Units address their own shank's electrode rows, by position."""

import unittest

import numpy as np
import pandas as pd

from nwb_zarr_smasher.nwb_zarr_smasher import (
    UNIT_ELECTRODE_GROUP_COL,
    electrode_group_for_unit,
    join_units_to_electrodes,
)

PROBE = "46116"
GROUPS = [f"{PROBE}_group{i}" for i in range(4)]
HYPHENATED = [f"{PROBE}-{i}" for i in range(1, 5)]


def _electrodes(groups, per_group=4, stride=2):
    """Electrodes as a multi-shank probe is packaged.

    Channel numbers are probe-global and their per-group ranges overlap, so a
    number identifies a row only together with its group -- and not even then
    in the same order the rows are stored. *stride* sets the overlap.
    """
    rows = []
    for index, group in enumerate(groups):
        for position in range(per_group):
            channel = index * stride + position
            rows.append(
                {
                    "group_name": group,
                    "channel_index": channel,
                    "channel_name": f"AP-CH{channel}",
                    "location": f"{group}:{position}",
                }
            )
    return pd.DataFrame(rows)


def _units(device, shank, channels):
    """Units keyed the way the sorted NWB writes them, shank-local channels."""
    return pd.DataFrame(
        {
            "device_name": device,
            "shank": shank,
            "extremum_channel_index": channels,
        }
    )


class ElectrodeGroupForUnitTest(unittest.TestCase):
    """The convention is read off the electrodes table, never assumed."""

    def test_device_that_is_already_a_group_is_kept(self):
        """The ``<probe>-<N>`` layout carries its shank in the device name."""
        self.assertEqual(
            electrode_group_for_unit(f"{PROBE}-1", 0, set(HYPHENATED)),
            f"{PROBE}-1",
        )

    def test_bare_device_gains_its_shank(self):
        """The breakage: ``46116`` + ``group1`` against ``46116_group1``."""
        self.assertEqual(
            electrode_group_for_unit(PROBE, "group1", set(GROUPS)),
            f"{PROBE}_group1",
        )

    def test_numeric_shank_reaches_a_hyphenated_group(self):
        """A 0-based shank number against the 1-based ``-<N>`` spelling."""
        self.assertEqual(
            electrode_group_for_unit(PROBE, 0, set(HYPHENATED)), f"{PROBE}-1"
        )

    def test_group_shank_reaches_a_hyphenated_group(self):
        """The two spellings are reconciled in either direction."""
        self.assertEqual(
            electrode_group_for_unit(PROBE, "group2", set(HYPHENATED)),
            f"{PROBE}-3",
        )

    def test_shank_read_from_a_float_column(self):
        """A numeric shank column hands over ``1.0``, not ``1``."""
        self.assertEqual(
            electrode_group_for_unit(PROBE, 1.0, set(GROUPS)),
            f"{PROBE}_group1",
        )

    def test_absent_shank_is_tolerated(self):
        """A single-group probe has no shank and needs no repair."""
        for shank in (None, np.nan, "", "  "):
            self.assertEqual(
                electrode_group_for_unit(PROBE, shank, {PROBE}), PROBE
            )

    def test_unknown_shank_never_invents_a_group(self):
        """An unmatched shank must surface as an empty join, not a guess."""
        self.assertEqual(
            electrode_group_for_unit(PROBE, "group9", {f"{PROBE}_group0"}),
            PROBE,
        )


class JoinUnitsToElectrodesTest(unittest.TestCase):
    """Both halves of the key are reconciled, or the join refuses."""

    def test_units_index_their_group_by_position(self):
        """The regression that assigns plausible, wrong regions.

        ``46116_group1`` holds channels 2..5, so a unit whose shank-local
        index is 3 belongs to the group's fourth row -- channel 5 -- not to
        the row that happens to be numbered 3.
        """
        merged = join_units_to_electrodes(
            _units(PROBE, ["group1"], [3]), _electrodes(GROUPS)
        )

        self.assertEqual(merged["location"].tolist(), [f"{PROBE}_group1:3"])
        self.assertNotIn(f"{PROBE}_group1:1", merged["location"].tolist())
        self.assertEqual(merged["channel_index"].tolist(), [5])

    def test_bare_device_reaches_its_own_shank(self):
        """Each shank's units take that shank's regions, not shank 0's."""
        units = _units(PROBE, [f"group{i}" for i in range(4)], [0, 1, 2, 3])

        merged = join_units_to_electrodes(units, _electrodes(GROUPS))

        self.assertEqual(
            merged["location"].tolist(),
            [f"{group}:{i}" for i, group in enumerate(GROUPS)],
        )
        self.assertEqual(merged[UNIT_ELECTRODE_GROUP_COL].tolist(), GROUPS)

    def test_contiguous_channel_numbering_is_unchanged(self):
        """Where position already equals the channel number, nothing moves.

        This is every probe that joins correctly today, and the positional key
        must reproduce it rather than shift those units by a row.
        """
        electrodes = _electrodes([PROBE])
        units = _units(PROBE, [np.nan, np.nan], [1, 3])

        merged = join_units_to_electrodes(units, electrodes)

        by_number = units.merge(
            electrodes,
            left_on=["device_name", "extremum_channel_index"],
            right_on=["group_name", "channel_index"],
            how="left",
        )
        self.assertEqual(
            merged["location"].tolist(), by_number["location"].tolist()
        )

    def test_units_keep_their_row_order(self):
        """Spike times stay attached to their unit, so order cannot shift."""
        units = _units(PROBE, ["group3", "group0", "group1"], [3, 2, 0])

        merged = join_units_to_electrodes(units, _electrodes(GROUPS))

        self.assertEqual(
            merged["location"].tolist(),
            [f"{PROBE}_group3:3", f"{PROBE}_group0:2", f"{PROBE}_group1:0"],
        )

    def test_the_callers_frames_are_not_modified(self):
        """Neither table is the join's to edit."""
        units = _units(PROBE, ["group1"], [1])
        electrodes = _electrodes(GROUPS)

        join_units_to_electrodes(units, electrodes)

        self.assertNotIn(UNIT_ELECTRODE_GROUP_COL, units.columns)
        self.assertNotIn("_pos_in_group", electrodes.columns)

    def test_units_without_a_shank_column_still_join(self):
        """The column is absent on probes that never needed it."""
        units = _units(PROBE, "", [2]).drop(columns="shank")

        merged = join_units_to_electrodes(units, _electrodes([PROBE]))

        self.assertEqual(merged["location"].tolist(), [f"{PROBE}:2"])

    def test_electrodes_without_channel_numbers_still_join(self):
        """Position is read off the row order; the number is only audited."""
        electrodes = _electrodes(GROUPS).drop(
            columns=["channel_index", "channel_name"]
        )

        merged = join_units_to_electrodes(
            _units(PROBE, ["group2"], [1]), electrodes
        )

        self.assertEqual(merged["location"].tolist(), [f"{PROBE}_group2:1"])

    def test_units_past_the_end_of_their_group_raise(self):
        """A unit indexing beyond its group's channels is not a blank row."""
        units = _units(PROBE, ["group1"] * 2, [4, 9])

        with self.assertRaises(ValueError) as caught:
            join_units_to_electrodes(units, _electrodes(GROUPS))

        message = str(caught.exception)
        self.assertIn("2 unit(s) matched no electrode row", message)
        self.assertIn(f"{PROBE}_group1: 2 unit(s) indexing 4..9", message)
        self.assertIn("the group has 4 channel(s)", message)

    def test_an_unresolved_shank_raises_and_names_the_probe(self):
        """The whole-probe loss this fix exists to stop, made loud."""
        units = _units(PROBE, ["group9"] * 3, [0, 1, 2])

        with self.assertRaises(ValueError) as caught:
            join_units_to_electrodes(units, _electrodes(GROUPS))

        message = str(caught.exception)
        self.assertIn(f"  {PROBE}: 3 unit(s)", message)
        self.assertIn("no electrode group of that name", message)

    def test_missing_join_columns_are_named(self):
        """A schema change must not degrade into an empty join."""
        with self.assertRaises(ValueError) as caught:
            join_units_to_electrodes(
                _units(PROBE, "", [0]).drop(columns="device_name"),
                _electrodes([PROBE]),
            )
        self.assertIn("device_name", str(caught.exception))

        with self.assertRaises(ValueError) as caught:
            join_units_to_electrodes(
                _units(PROBE, "", [0]),
                _electrodes([PROBE]).drop(columns="group_name"),
            )
        self.assertIn("group_name", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
