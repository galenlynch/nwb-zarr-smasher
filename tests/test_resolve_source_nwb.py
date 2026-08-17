"""The source NWB is found by glob, not by rebuilding the asset's name."""

import tempfile
import unittest
from pathlib import Path

from nwb_zarr_smasher.nwb_zarr_smasher import resolve_source_nwb

SORTED = "ecephys_771432_2025-05-07_18-22-06_sorted_2026-07-30_11-52-00"
ACQUIRED = "ecephys_771432_2025-03-07_18-22-06"
UPLOADED = "ecephys_771432_2025-05-07_18-22-06"


class ResolveSourceNwbTest(unittest.TestCase):
    """Cover the date-defect case and the fallbacks around it."""

    def setUp(self):
        """Give each test its own mount root."""
        self._tmp = tempfile.TemporaryDirectory()
        self.data_folder = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _nwb_dir(self, sorted_name=SORTED):
        """Create and return the sorted asset's nwb/ directory."""
        nwb_dir = self.data_folder / sorted_name / "nwb"
        nwb_dir.mkdir(parents=True)
        return nwb_dir

    def test_globs_past_a_date_defective_asset_name(self):
        """Asset named for upload date, NWBs named for acquisition date."""
        nwb_dir = self._nwb_dir()
        for exp in (1, 2, 3):
            (nwb_dir / f"{ACQUIRED}_experiment{exp}_recording1.nwb").touch()

        self.assertEqual(
            resolve_source_nwb(self.data_folder, SORTED, 1, 1),
            nwb_dir / f"{ACQUIRED}_experiment1_recording1.nwb",
        )
        self.assertEqual(
            resolve_source_nwb(self.data_folder, SORTED, 3, 1),
            nwb_dir / f"{ACQUIRED}_experiment3_recording1.nwb",
        )

    def test_falls_back_to_the_reconstructed_name_without_an_nwb_dir(self):
        """A miss must still name the path that was expected."""
        self.assertEqual(
            resolve_source_nwb(self.data_folder, SORTED, 1, 1),
            self.data_folder
            / SORTED
            / "nwb"
            / f"{UPLOADED}_experiment1_recording1.nwb",
        )

    def test_falls_back_when_the_experiment_is_absent(self):
        """Only experiment1 exists; asking for 2 reports the expected name."""
        nwb_dir = self._nwb_dir()
        (nwb_dir / f"{ACQUIRED}_experiment1_recording1.nwb").touch()

        self.assertEqual(
            resolve_source_nwb(self.data_folder, SORTED, 2, 1),
            nwb_dir / f"{UPLOADED}_experiment2_recording1.nwb",
        )

    def test_prefers_the_reconstructed_name_when_several_match(self):
        """The name matching the asset wins; no exception."""
        nwb_dir = self._nwb_dir()
        (nwb_dir / f"{UPLOADED}_experiment1_recording1.nwb").touch()
        (nwb_dir / f"{ACQUIRED}_experiment1_recording1.nwb").touch()

        self.assertEqual(
            resolve_source_nwb(self.data_folder, SORTED, 1, 1),
            nwb_dir / f"{UPLOADED}_experiment1_recording1.nwb",
        )

    def test_raises_when_several_match_and_none_is_the_asset_name(self):
        """Guard the glob: never silently pick one of several strangers."""
        nwb_dir = self._nwb_dir()
        (nwb_dir / "a_experiment1_recording1.nwb").touch()
        (nwb_dir / "b_experiment1_recording1.nwb").touch()

        with self.assertRaisesRegex(ValueError, "Several NWBs"):
            resolve_source_nwb(self.data_folder, SORTED, 1, 1)


if __name__ == "__main__":
    unittest.main()
