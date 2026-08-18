"""A multi-shank probe folder maps onto one electrode group per shank."""

import unittest

from nwb_zarr_smasher.nwb_zarr_smasher import electrode_group_for_shank

PROBE = "46116"


class ElectrodeGroupForShankTest(unittest.TestCase):
    """The convention is read off the table, never assumed."""

    def test_bare_file_is_the_probe_folder(self):
        """A single-shank stream writes no suffix and needs no mapping."""
        self.assertEqual(
            electrode_group_for_shank(
                PROBE, "ccf_channel_locations.json", {PROBE}
            ),
            PROBE,
        )

    def test_split_by_group_naming(self):
        """spikeinterface split_by("group") gives 0-based ``_group`` names."""
        groups = {f"{PROBE}_group{i}" for i in range(4)}
        for shank, expected in enumerate(sorted(groups), start=1):
            self.assertEqual(
                electrode_group_for_shank(
                    PROBE, f"ccf_channel_locations_shank{shank}.json", groups
                ),
                expected,
            )

    def test_hyphenated_naming(self):
        """Other sorts name the same shanks ``<probe>-<N>``, 1-based."""
        groups = {f"{PROBE}-{i}" for i in range(1, 5)}
        self.assertEqual(
            electrode_group_for_shank(
                PROBE, "ccf_channel_locations_shank3.json", groups
            ),
            f"{PROBE}-3",
        )

    def test_unknown_shank_falls_back_to_the_probe_folder(self):
        """An unmatched shank joins to nothing, never to an invented group."""
        self.assertEqual(
            electrode_group_for_shank(
                PROBE, "ccf_channel_locations_shank2.json", {"45883-1"}
            ),
            PROBE,
        )

    def test_group_naming_wins_when_both_conventions_are_present(self):
        """Ambiguity resolves deterministically instead of by set order."""
        groups = {f"{PROBE}_group0", f"{PROBE}-1"}
        self.assertEqual(
            electrode_group_for_shank(
                PROBE, "ccf_channel_locations_shank1.json", groups
            ),
            f"{PROBE}_group0",
        )


if __name__ == "__main__":
    unittest.main()
