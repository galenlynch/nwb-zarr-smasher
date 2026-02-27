# nwb-zarr-smasher

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-100.0%25-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![Python](https://img.shields.io/badge/python->=3.10-blue?logo=python)

## Usage
Minimal usage: smash_nwb

The smash_nwb function provides a one-shot workflow to:

Convert Bonsai behavior JSON → behavior NWB

Merge behavior into a sorted ephys NWB/Zarr

Inject electrode annotations from IBL app output

(Optional) Align timestamps using HARP

(Optional) Add sniff detector acquisition

Example
```
from nwb_zarr_smasher.smash import smash_nwb

out_path = smash_nwb(
    ephys_sorted_name="ecephys_XXXX_YYYY_sorted_ZZZZ",
    beh_json="/data/path/to/behavior.json",
    ibl_app_output="/results/path/to/ibl_app_output",

    # optional
    do_harp_alignment=False,
    harp_channel=5,
    sniffing_folder=None,
)
print("Merged NWB written to:", out_path)
```
Required inputs

ephys_sorted_name:	Name of the sorted ephys dataset folder under /data

beh_json:	Path to Bonsai behavior JSON

ibl_app_output:	Directory containing IBL app probe annotations


Optional inputs
do_harp_alignment:Run HARP timestamp realignment

harp_channel:Digital line used for HARP clock

sniffing_folder:Folder containing SniffDetector__32.bin files (only used if provided)


## Installation
To use the software, in the root directory, run
```bash
pip install -e .
```

To develop the code, run
```bash
pip install -e . --group dev
```
Note: --group flag is available only in pip versions >=25.1

Alternatively, if using `uv`, run
```bash
uv sync
```
