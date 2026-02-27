# Some Chat-GPT code to merge behavior and file into new ephys zarr


import zarr
from hdmf_zarr import NWBZarrIO
from pynwb import TimeSeries
from pynwb.behavior import (
    BehavioralTimeSeries,
    EyeTracking,
    PupilTracking,
    SpatialSeries
)
from pynwb import  NWBHDF5IO,NWBFile
from pathlib import Path
import pandas as pd
import numpy as np
import utils
import shutil

import aind_dynamic_foraging_data_utils.nwb_utils as nwb_utils
from aind_ephys_utils.align import align_to_events

from matplotlib import pyplot as plt
import ast

import datetime
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt 

from typing import Optional, Dict, Tuple,Union



import os, shutil, numpy as np
from pynwb.core import ScratchData
from pynwb.file import Subject
from pynwb.file import ElectrodeTable

import json
from pynwb.core import DynamicTable, VectorData
from pynwb import TimeSeries


from open_ephys.analysis import Session
# Older harp
from harp.clock import align_timestamps_to_anchor_points, decode_harp_clock

# 

from matplotlib import pyplot as plt

from iblatlas.atlas import AllenAtlas
import SimpleITK as sitk

from hdmf_zarr import NWBZarrIO


# --- IO helpers (Zarr vs HDF5, and clean copy) ---
def _norm_mode(mode: str) -> str:
    return 'r+' if mode == 'a' else mode

def _pick_io(path: str, mode: str):
    mode = _norm_mode(mode)
    if os.path.isdir(path):
        from hdmf_zarr.nwb import NWBZarrIO
        return NWBZarrIO(path, mode=mode)          # Zarr directory
    else:
        return NWBHDF5IO(path, mode=mode)          # HDF5 file

def _copy_src_to_dst(src: str, dst: str):
    # Always start from a clean destination to avoid stale/invalid metadata
    if os.path.exists(dst):
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        else:
            os.remove(dst)
    if os.path.isdir(src):
        shutil.copytree(src, dst)     # fresh clone of Zarr dir
    else:
        shutil.copy2(src, dst)        # HDF5 file copy

# --- string & scratch sanitizers (avoid NaN/None for vlen-UTF8) ---
def _is_string_column(vec) -> bool:
    try:
        data = np.asarray(vec.data)
    except Exception:
        return False
    if data.dtype.kind in ('U','S'):
        return True
    if data.dtype.kind == 'O':
        # find a non-missing example
        for v in np.ravel(data):
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                return isinstance(v, str)
    return False

def _coerce_value_for_column(vec, val):
    if _is_string_column(vec):
        if val is None:
            return ""
        if isinstance(val, float) and np.isnan(val):
            return ""
        return val if isinstance(val, str) else ("" if val is None else str(val))
    return val

def _clean_string_array_for_zarr(arr):
    a = np.asarray(arr, dtype=object)
    with np.errstate(invalid='ignore'):
        nan_mask = np.vectorize(lambda x: isinstance(x, float) and np.isnan(x))(a)
    none_mask = (a == None)  # noqa: E711
    a[nan_mask | none_mask] = ''
    return a.astype('U')

def _merge_top_metadata(dst, beh):
    """Copy a few useful NWBFile-level fields if absent on dst."""
    if getattr(beh, "protocol", None) and not getattr(dst, "protocol", None):
        dst.protocol = beh.protocol
    if getattr(beh, "experiment_description", None) and not getattr(dst, "experiment_description", None):
        dst.experiment_description = beh.experiment_description
    # merge keywords (set-union)
    try:
        beh_kw = set(beh.keywords or [])
        dst_kw = set(dst.keywords or [])
        merged = sorted(dst_kw | beh_kw)
        if merged:
            dst.keywords = merged
    except Exception:
        pass  # keywords may be None/unsupported in some versions

def _copy_behavior_acquisitions(dst, beh):
    """
    Copy all root-level acquisitions from 'beh' into 'dst'.
    - Try a fast deepcopy (works for most TimeSeries).
    - On failure, reconstruct a minimal TimeSeries to avoid parentage errors.
    - Deduplicate names if needed.
    """
    for name, obj in beh.acquisition.items():
        new_name = name
        k = 1
        while new_name in dst.acquisition:
            k += 1
            new_name = f"{name}_{k}"

        # 1) try safe deepcopy
        try:
            clone = copy.deepcopy(obj)
            clone.name = new_name
            dst.add_acquisition(clone)
            continue
        except Exception:
            pass

        # 2) minimal fallback for TimeSeries-like acquisitions
        if isinstance(obj, TimeSeries):
            ts_kwargs = dict(
                name=new_name,
                data=obj.data,  # will stream on write; no full RAM copy
                unit=getattr(obj, "unit", None),
                comments=getattr(obj, "comments", None),
                description=getattr(obj, "description", None),
                conversion=getattr(obj, "conversion", 1.0),
                resolution=getattr(obj, "resolution", np.nan),
            )
            # timestamps vs starting_time/rate
            if getattr(obj, "timestamps", None) is not None:
                ts_kwargs["timestamps"] = obj.timestamps
            else:
                ts_kwargs["starting_time"] = getattr(obj, "starting_time", 0.0)
                ts_kwargs["rate"] = getattr(obj, "rate", None)

            dst.add_acquisition(TimeSeries(**ts_kwargs))
        # else: if you have non-TimeSeries acquisitions you want copied,
        # tell me their types and I’ll add constructors for them.

# --- append behavior content (trials + scratch) ---
def _append_behavior_trials_and_scratch(dst, beh):
    # Trials: add only behavior columns; coerce strings’ missings -> ''
    beh_trials = getattr(beh, "trials", None)
    if beh_trials is not None and len(beh_trials) > 0:
        required = {"start_time", "stop_time", "id"}
        for col in beh_trials.colnames:
            if col in required:
                continue
            if getattr(dst, "trials", None) is None or col not in dst.trials.colnames:
                dst.add_trial_column(col, description=getattr(beh_trials[col], "description", None))
        for i in range(len(beh_trials)):
            row = {
                "start_time": float(beh_trials["start_time"][i]),
                "stop_time":  float(beh_trials["stop_time"][i]),
            }
            for col in beh_trials.colnames:
                if col in required:
                    continue
                vec = beh_trials[col]
                row[col] = _coerce_value_for_column(vec, vec[i])
            dst.add_trial(**row)

    # Scratch: build proper ScratchData containers; ensure UTF-8 clean
    beh_scratch = getattr(beh, "scratch", None)
    if beh_scratch:
        existing_names = set(sd.name for sd in (dst.scratch or []))
        for key, item in beh_scratch.items():
            name = key
            k = 1
            while name in existing_names:
                k += 1
                name = f"{key}_{k}"
            existing_names.add(name)

            # extract payload + description
            desc = getattr(item, "description", None)
            payload = item.data if hasattr(item, "data") else item
            try:
                arr = np.asarray(payload)
            except Exception:
                arr = np.asarray(str(payload))

            # sanitize possible string/object arrays
            if arr.dtype.kind in ('U','S','O'):
                arr = _clean_string_array_for_zarr(arr)

            if not desc:
                desc = f"Copied from behavior.scratch['{key}']"

            dst.add_scratch(ScratchData(name=name, data=arr, description=desc))

def _is_missing(v):
    if v is None: 
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    return isinstance(v, float) and math.isnan(v)

def _merge_subject_into_dst(dst, beh, prefer="ephys",
                            stash_unknown_to_scratch=True,
                            scratch_name="subject_merged_extras"):
    """
    Merge dst.subject (ephys) with beh.subject (behavior).
    Conflict policy: prefer 'ephys' (default) or 'behavior'.
    - If dst.subject exists: update fields in place (no reassignment).
    - If dst.subject is None: create a new Subject from merged values.
    - Unknown/custom attributes are serialized to scratch JSON.
    """
    sub_dst = getattr(dst, "subject", None)
    sub_beh = getattr(beh, "subject", None)
    if sub_dst is None and sub_beh is None:
        return

    # Which side do we prefer on conflicts?
    preferred, other = (sub_dst, sub_beh) if prefer == "ephys" else (sub_beh, sub_dst)

    # Standard Subject fields we’ll merge
    fields = [
        "subject_id", "description", "age", "species",
        "genotype", "sex", "weight", "date_of_birth",
        "strain", "phenotype",
    ]

    def pick(key):
        """Choose a value for a single field based on policy."""
        v_pref = getattr(preferred, key, None) if preferred is not None else None
        v_other = getattr(other, key, None) if other is not None else None
        if prefer == "ephys":
            return v_pref if not _is_missing(v_pref) else (v_other if not _is_missing(v_other) else v_pref)
        else:
            return v_other if not _is_missing(v_other) else (v_pref if not _is_missing(v_pref) else v_other)

    merged = {k: pick(k) for k in fields if not _is_missing(pick(k))}

    # Write back: update in place if dst.subject exists; else construct fresh
    if sub_dst is None:
        dst.subject = Subject(**merged)
    else:
        for k, v in merged.items():
            try:
                setattr(sub_dst, k, v)
            except Exception:
                # Some fields may be read-only in older versions; skip quietly
                pass

    # Optional: stash non-standard attributes from both subjects to scratch
    if stash_unknown_to_scratch:
        extras = {}
        for label, s in (("preferred", preferred), ("other", other)):
            if s is None:
                continue
            for k, v in vars(s).items():
                if k.startswith("_") or k in fields or k in {"container_source", "parent", "name"}:
                    continue
                # keep preferred’s value unless missing
                if label == "preferred":
                    extras.setdefault(k, v)
                else:
                    if k not in extras or _is_missing(extras[k]):
                        extras[k] = v

        if extras:
            def to_jsonable(x):
                try:
                    json.dumps(x)
                    return x
                except Exception:
                    return str(x)

            cleaned = {k: to_jsonable(v) for k, v in extras.items()}
            payload = np.array([json.dumps(cleaned, default=str)], dtype="U")
            desc = f"Merged non-standard Subject fields (prefer={prefer})."
            existing = set(sd.name for sd in (dst.scratch or []))
            name = scratch_name
            i = 1
            while name in existing:
                i += 1
                name = f"{scratch_name}_{i}"
            dst.add_scratch(ScratchData(name=name, data=payload, description=desc))

def merge_behavior_into_ephys(ephys_path: str, behavior_path: str, out_path: str):
    # 0) fresh copy of the ephys NWB (file or Zarr dir)
    _copy_src_to_dst(ephys_path, out_path)

    # 1) open copied ephys in r+, behavior in r, with proper IOs
    with _pick_io(out_path, mode='r+') as out_io, _pick_io(behavior_path, mode='r') as beh_io:
        dst = out_io.read()
        beh = beh_io.read()

        _append_behavior_trials_and_scratch(dst, beh)
        _merge_top_metadata(dst, beh)
        _copy_behavior_acquisitions(dst, beh)
        _merge_subject_into_dst(dst,beh,prefer = 'ephys')
        
        out_io.write(dst)
        return out_io

def harp_timestamp_salvage(directory,
                           harp_line = 5,
                           main_event_line = 1,
                           event_stream_name = 'PXIe-6341',
                           output_parent = '/scratch/'):
    """
    Extremely good description
    """
    
    session = Session(directory)
    
    for recordnode_index in range(len(session.recordnodes)):
        events = session.recordnodes[recordnode_index].recordings[0].events
    
        main_events = events[(events.stream_name == event_stream_name) &
                             (events.line == main_event_line) &
                             (events.state == 1)]
        
        harp_events = events[(events.stream_name == event_stream_name) &
                             (events.line == harp_line)]
        
        start_times, harp_times = decode_harp_clock(harp_events.sample_number.values / 30000, 
                    harp_events.state.values) 
        
        for stream in session.recordnodes[recordnode_index].recordings[0].continuous:
        
            plt.clf()
        
            stream_name = stream.metadata['stream_name']
        
            print(stream_name)
        
            sync_events = events[(events.stream_name == stream_name) &
                                (events.line == 1) &
                                (events.state == 1)]
            print(f'Num sync events: {len(sync_events)}')
            print(f'Num main events: {len(main_events)}')
            print(main_events.state.values[:10])
            print(sync_events.state.values[:10])
        
            local_aligned_timestamps = align_timestamps_to_anchor_points(stream.sample_numbers,
                sync_events.sample_number.values,
                main_events.sample_number.values / 30000)
        
            harp_aligned_timestamps = align_timestamps_to_anchor_points(local_aligned_timestamps, 
                        start_times,
                        harp_times) 
        
            output_directory = os.path.join(output_parent,stream_name)
            os.makedirs(output_directory, exist_ok=True)
            
            np.save(output_directory + '/timestamps.npy', harp_aligned_timestamps)
        
            plt.subplot(1,2,1)
            plt.hist(np.diff(local_aligned_timestamps), bins=np.arange(-0.0001,0.0001,0.00001))
            plt.title('Local timestamp diff')
        
            plt.subplot(1,2,2)
            plt.hist(np.diff(harp_aligned_timestamps), bins=np.arange(-0.0001,0.0001,0.00001))
            plt.title('Harp timestamp diff')
        
            print(harp_aligned_timestamps[:10])
        
            plt.savefig(output_directory + '/timestamp_diff.png')

def update_units_table_timestamps(nwb_output_path,ephys_sorted_name,temp_harp_dir = '/scratch/',experiment = 'experiment1',recording = 'recording1',data_dir = '/data/',):
    """
    Blah Blah Blah
    """
    io = NWBZarrIO(str(nwb_output_path), "r+")
    ephys_nwb = io.read()
    units = ephys_nwb.units.to_dataframe()
    
    # Get the harp-corrected unit timestamps
    look_up = {'1':'-1','1-1':'-2','1-2':'-3','1-3':'-4'}
    
    # Fix the the units
    unq_devices = np.unique(units.device_name)
    for ii,device in enumerate(unq_devices):
        print(device)
    
        if '-' in device:
            tmp_device = device.split('-')[0]+look_up[device.split('-')[1]]
        else:
            tmp_device = device
    
        folder_name = [x for x in os.listdir(os.path.join(data_dir,f'{ephys_sorted_name}/spikesorted/')) if ((tmp_device in x) & (experiment in x) & (recording in x))][0]
        
        spikes = np.load(os.path.join(data_dir,f'{ephys_sorted_name}/spikesorted/{folder_name}/spikes.npy'))
    
    
        fixed_timestamps = np.load(os.path.join(temp_harp_dir,f'{tmp_device}/timestamps.npy'))
    
        
        spk_index = np.array([x[0] for x in spikes])
        spk_id = np.array([x[1] for x in spikes])
        unq_units = np.unique(spk_id)
        spk_ts = {}
        
        for ii,uu in enumerate(unq_units):
            mask = (units['device_name'] == device) & (units['ks_unit_id'] == uu)
            if np.any(mask):
                idx = units.index[mask][0]   # first matching row
                units.at[idx, 'spike_times'] = fixed_timestamps[spk_index[spk_id == uu]]
    return units

def write_new_unit_timestamps_to_file(nwb_path,new_units_table):
    """
    Blah Blah BLah
    """
    # New spike times come from new units table
    spike_times_list = new_units_table['spike_times'].tolist()
    
    # Open zarr, allow read right
    zarr_root = zarr.open(nwb_path, mode="a")
    
    # Nab data type from old zarr ts.
    old_spike_times = zarr_root["units"]["spike_times"]
    spike_times_dtype = old_spike_times.attrs["zarr_dtype"]

    # Write new spike times to existing zarr
    new_spike_times = np.concatenate(spike_times_list)
    zarr_root["units"]["spike_times"][:] = new_spike_times.astype(spike_times_dtype)




# Merge electrodes table with IBL app output
def update_electrodes_table_locations_from_ibl_app(nwb_output_path,ibl_annotations_path):
    """
    Moving Prose
    """
    
    # Read copied NWB
    io = NWBZarrIO(str(nwb_output_path), "r+")
    ephys_nwb = io.read()
    # Get the electrodes table
    electrodes = ephys_nwb.electrodes.to_dataframe()

    electrode_names = os.listdir(ibl_annotations_path)

    group_name = []
    channel_name = []
    brain_region_id = []
    location = []
    ccf_ml = []
    ccf_ap = []
    ccf_dv = []
    
    atlas = None
    
    for ii,elect in enumerate(electrode_names):
        try: # Histology space file format
            with open(os.path.join(ibl_annotations_path,elect,'ccf_channel_locations.json'),'r') as O:
                ccf_json = json.load(O)
    
            for jj,channel_id in enumerate(ccf_json.keys()):
                group_name.append(elect)
                channel_name.append('CH'+channel_id.split('_')[-1])
                brain_region_id.append(ccf_json[channel_id]['brain_region_id'])
                location.append(ccf_json[channel_id]['brain_region'])
                
                #ccf_mlapdv = np.array([ccf_json[channel_id]['x'],ccf_json[channel_id]['y'],ccf_json[channel_id]['z']])*1000
                ccf_ml.append(ccf_json[channel_id]['x']*1000)
                ccf_ap.append(ccf_json[channel_id]['y']*1000)
                ccf_dv.append(ccf_json[channel_id]['z']*1000)
        except FileNotFoundError:
            if not atlas:
                atlas = AllenAtlas(mock = True)
                ccf_annotations = sitk.ReadImage('/data/allen_mouse_ccf/annotation/ccf_2017/annotation_10.nii.gz')
                areas = pd.read_csv('/data/allen_mouse_ccf/annotation/adult_mouse_ccf_structures.csv')
            try:
                with open(os.path.join(ibl_annotations_path,elect,'channel_locations.json'),'r') as O:
                    ccf_json = json.load(O)
            except FileNotFoundError:
                continue
            for jj,channel_id in enumerate(ccf_json.keys()):
                if 'origin' in channel_id:
                    continue
                group_name.append(elect)
                channel_name.append('CH'+channel_id.split('_')[-1])
    
                try:
                    ccf_mlapdv = atlas.xyz2ccf(np.array([ccf_json[channel_id]['x'],ccf_json[channel_id]['y'],ccf_json[channel_id]['z']]),ccf_order='mlapdv')         
                    #ccf_mlapdv = np.array([ccf_json[channel_id]['x'],ccf_json[channel_id]['y'],ccf_json[channel_id]['z']])*1000
                except ValueError:
                    ccf_mlapdv = [0,0,0]
                ccf_ml.append(-ccf_mlapdv[0])
                ccf_ap.append(ccf_mlapdv[1])
                ccf_dv.append(-ccf_mlapdv[2])
    
                ccf_mm = np.array([-ccf_mlapdv[0],ccf_mlapdv[1],-ccf_mlapdv[2]])/1000
                brain_region_number = ccf_annotations.GetPixel(ccf_annotations.TransformPhysicalPointToIndex(ccf_mm))
                brain_region_id.append(brain_region_number)
                try:
                    location.append(areas.acronym.values[list(areas.id.values).index(brain_region_number)])
                except ValueError:
                    location.append(str(np.nan))
    
            
    df = pd.DataFrame({'group_name':group_name,
                       'channel_name':channel_name,
                       'brain_region_id':brain_region_id,
                       'brain_region':location,
                      'ccf_ml':ccf_ml,
                      'ccf_ap':ccf_ap,
                      'ccf_dv':ccf_dv,
    
                      })
    
    # # Rename electrodes to match the units ephys
    # F = {'-1-1':'-2','-1-2':'-3','-1-3':'-4',}
    # for xx,findkey in enumerate(F.keys()):
    #     chngkey = F[findkey]
    #     for ii,isin in enumerate([findkey in x for x in df.group_name]):
    #         if isin:
    #             this_loc = df.group_name.loc[ii]
    #             df.group_name.loc[ii] = this_loc.split(findkey)[0]+chngkey
    
    merged = (
        pd.merge(
            electrodes,
            df,
            on=["group_name", "channel_name"],
            how="left",
            suffixes=("_electrodes", "_df")
        )
        .drop(columns=["location"], errors="ignore")   # remove old "location" if present
        .rename(columns={"brain_region": "location"})  # rename brain_region → location
    )

    return merged


# Some ChatGPT code to save the new electrodes table
def _safe_location(row, group_obj):
    # 1) if merged has a non-empty location, use it
    if "location" in row.index:
        val = row["location"]
        if pd.notna(val):
            s = str(val).strip()
            if s and s.lower() != "nan":
                return s
    # 2) else try the group's location if available and non-empty
    try:
        if getattr(group_obj, "location", None):
            s = str(group_obj.location).strip()
            if s:
                return s
    except Exception:
        pass
    # 3) final fallback: group name (always non-empty)
    return str(group_obj.name)


def replace_electrodes_table_with_merged(nwb_path: str,
                                         merged: pd.DataFrame,
                                         *,
                                         create_missing_groups: bool = False) -> None:
    """
    Completely replace the electrodes table with rows/columns from `merged`.

    Requirements:
      - `merged` must include 'group_name' (used to look up ElectrodeGroup).
      - All other columns are written as-is (except 'group_name' and 'group').
      - If 'impedance' is present (but 'imp' not), it is mapped to NWB's 'imp'.
      - If 'locations' exists (but not 'location'), it is renamed to 'location'.

    This function:
      1) Removes the old table (nwb.fields.pop('electrodes', None)).
      2) Pre-declares dynamic columns on a fresh empty table.
      3) Adds each row via `nwb.add_electrode(...)`.
    """
    if "group_name" not in merged.columns:
        raise ValueError("`merged` must contain a 'group_name' column.")

    # Common alias fix
    if "location" not in merged.columns and "locations" in merged.columns:
        merged = merged.rename(columns={"locations": "location"})

    reserved = {"group", "group_name"}
    std_optional = ("x", "y", "z", "imp", "impedance", "location", "filtering")

    with NWBZarrIO(nwb_path, "r+") as io:
        nwb = io.read()

        # Ensure ElectrodeGroups exist (or create if requested)
        gmap = {g.name: g for g in nwb.electrode_groups.values()}
        missing = sorted(set(merged["group_name"].unique()) - set(gmap.keys()))
        if missing:
            if not create_missing_groups:
                raise ValueError(f"Missing ElectrodeGroup(s) in NWB: {missing}")
            device = next(iter(nwb.devices.values()), None) or nwb.create_device("device0")
            for name in missing:
                gmap[name] = nwb.create_electrode_group(
                    name=name, description=f"Auto-created group {name}", device=device, location=""
                )

        # --- 1) Remove the existing electrodes table for real ---
        nwb.fields.pop('electrodes', None)  # hard reset

        # --- 2) Predeclare dynamic columns on a fresh empty table
        dynamic_cols = [c for c in merged.columns if c not in reserved and c not in std_optional]
        for col in dynamic_cols:
            # This implicitly creates a new empty ElectrodeTable, since it was just removed
            nwb.add_electrode_column(name=col, description=f"{col} from merged")

        # --- 3) Add rows (must provide keys for *all* known columns each time)
        for _, row in merged.iterrows():
            group_obj = gmap[str(row["group_name"])]

            row_kwargs = {c: row[c] for c in dynamic_cols}  # dynamic columns

            # Standard optionals (always provide)
            row_kwargs["x"] = float(row["x"]) if "x" in merged.columns and pd.notna(row["x"]) else np.nan
            row_kwargs["y"] = float(row["y"]) if "y" in merged.columns and pd.notna(row["y"]) else np.nan
            row_kwargs["z"] = float(row["z"]) if "z" in merged.columns and pd.notna(row["z"]) else np.nan
            
            if "imp" in merged.columns and pd.notna(row.get("imp")):
                row_kwargs["imp"] = float(row["imp"])
            elif "impedance" in merged.columns and pd.notna(row.get("impedance")):
                row_kwargs["imp"] = float(row["impedance"])
            else:
                row_kwargs["imp"] = np.nan
            
            row_kwargs["location"]  = _safe_location(row, group_obj)   # <-- required, non-empty
            row_kwargs["filtering"] = str(row["filtering"]) if "filtering" in merged.columns and pd.notna(row.get("filtering")) else ""

            nwb.add_electrode(group=group_obj, **row_kwargs)

        io.write(nwb)

def add_units_locations_from_electrodes_zarr(
    nwb_zarr_path: str,
    *,
    unit_device_col: str = "device_name",
    unit_channel_col: str = "extremum_channel_index",
    elec_group_col: str = "group_name",
    elec_channel_index_col: str = "channel_index",
    elec_channel_name_col: str = "channel_name",
    cols_to_add: Tuple[str, ...] = ("location", "ccf_ml", "ccf_ap", "ccf_dv"),
    descriptions: Optional[Dict[str, str]] = None,
    overwrite_existing: bool = True,
) -> pd.DataFrame:
    """
    Add electrode-derived columns (location/CCF coords) into the NWB units table (Zarr).

    Join key:
      units[unit_device_col] + units[unit_channel_col]
        <-> electrodes[elec_group_col] + electrodes[elec_channel_index_col]
    """
    if descriptions is None:
        descriptions = {
            "location": "Electrode location (from electrodes table merge)",
            "ccf_ml": "CCF coordinate ML in microns (from electrodes table merge)",
            "ccf_ap": "CCF coordinate AP in microns (from electrodes table merge)",
            "ccf_dv": "CCF coordinate DV in microns (from electrodes table merge)",
        }

    def _to_str_no_nan(x) -> str:
        if x is None:
            return ""
        if isinstance(x, float) and np.isnan(x):
            return ""
        s = str(x).strip()
        return "" if s == "" or s.lower() == "nan" else s

    with NWBZarrIO(nwb_zarr_path, mode="r+") as io:
        nwb = io.read()

        if nwb.units is None:
            raise ValueError("This NWB has no units table (nwb.units is None).")
        if nwb.electrodes is None:
            raise ValueError("This NWB has no electrodes table (nwb.electrodes is None).")

        units_df = nwb.units.to_dataframe().copy()
        elec_df = nwb.electrodes.to_dataframe().copy()

        # --- validate join columns ---
        for col in (unit_device_col, unit_channel_col):
            if col not in units_df.columns:
                raise ValueError("units is missing required column: '%s'" % col)
        if elec_group_col not in elec_df.columns:
            raise ValueError("electrodes is missing required column: '%s'" % elec_group_col)

        # --- derive channel_index on electrodes if needed ---
        if elec_channel_index_col not in elec_df.columns:
            if elec_channel_name_col not in elec_df.columns:
                raise ValueError(
                    "electrodes is missing '%s' and cannot derive it (also missing '%s')."
                    % (elec_channel_index_col, elec_channel_name_col)
                )
            elec_df[elec_channel_index_col] = [
                int(str(x).split("CH")[-1]) for x in elec_df[elec_channel_name_col].values
            ]

        # --- normalize join-key types ---
        units_df[unit_device_col] = units_df[unit_device_col].astype(str)
        units_df[unit_channel_col] = pd.to_numeric(units_df[unit_channel_col], errors="coerce").astype("Int64")

        elec_df[elec_group_col] = elec_df[elec_group_col].astype(str)
        elec_df[elec_channel_index_col] = pd.to_numeric(elec_df[elec_channel_index_col], errors="coerce").astype("Int64")

        # --- preserve unit order robustly ---
        units_df = units_df.copy()
        units_df["__unit_row_index__"] = np.arange(len(units_df), dtype=np.int64)

        merged = pd.merge(
            units_df,
            elec_df,
            left_on=[unit_device_col, unit_channel_col],
            right_on=[elec_group_col, elec_channel_index_col],
            how="left",
            suffixes=("_units", "_elec"),
        )

        merged = merged.sort_values("__unit_row_index__", kind="stable")
        n_units = len(merged)

        def _prep_col(col: str):
            if col not in merged.columns:
                raise ValueError("After merge, column '%s' not found in merged DataFrame." % col)
            if col == "location":
                vals = merged[col].map(_to_str_no_nan).to_numpy(dtype="U")
            else:
                vals = pd.to_numeric(merged[col], errors="coerce").to_numpy(dtype=float)
            if len(vals) != n_units:
                raise RuntimeError("Column '%s' length mismatch." % col)
            return vals

        # --- add/overwrite NWB units columns ---
        for col in cols_to_add:
            vals = _prep_col(col)
            desc = descriptions.get(col, "%s (from electrodes table merge)" % col)

            if col not in nwb.units.colnames:
                nwb.units.add_column(name=col, description=desc, data=vals)
            else:
                if not overwrite_existing:
                    continue
                # Overwrite in-place if supported
                try:
                    nwb.units[col].data[:] = vals
                except Exception:
                    # element-wise fallback
                    for i in range(n_units):
                        nwb.units[col].data[i] = vals[i]

        io.write(nwb)

    # convenience return (units columns + requested new cols)
    out = merged.drop(columns=["__unit_row_index__"], errors="ignore")
    # keep original units col order first, then add cols_to_add if present
    out_cols = [c for c in units_df.columns if c != "__unit_row_index__" and c in out.columns]
    for c in cols_to_add:
        if c in out.columns and c not in out_cols:
            out_cols.append(c)
    return out[out_cols]

def smash_nwb(
    ephys_sorted_name: str,
    beh_json: Union[str, os.PathLike],
    ibl_app_output: Union[str, os.PathLike],
    *,
    do_harp_alignment: bool = False,
    harp_channel: int = 5,
    sniffing_folder: Optional[Union[str, os.PathLike]] = None,
    # sensible defaults for CodeOcean/AIND layouts (match your notebook)
    data_folder: Union[str, os.PathLike] = "/data",
    results_folder: Union[str, os.PathLike] = "/results",
    output_parent_tag: str = "nwb",
    scratch_dir: Union[str, os.PathLike] = "/scratch",
    experiment: int = 1,
    recording: int = 1,
    copy_metadata_jsons: bool = True,
) -> Path:
    """
    One-shot wrapper for the notebook workflow:
      1) bonsai JSON -> behavior NWB (scratch)
      2) merge behavior into ephys NWB/Zarr copy in /results
      3) update electrodes from IBL app output; add location/CCF columns to units
      4) (optional) do HARP alignment and rewrite units.spike_times
      5) (optional) add sniff detector acquisition from HARP SniffDetector__32.bin files

    Parameters
    ----------
    ephys_sorted_name
        Name of the sorted ephys folder under /data, e.g.
        "ecephys_XXXX_..._sorted_YYYY-...".
    beh_json
        Path to the bonsai behavior JSON.
    ibl_app_output
        Path to IBL app output directory that contains per-probe annotations folders.
    do_harp_alignment
        If True, run harp timestamp salvage and rewrite units.spike_times.
    harp_channel
        Digital line used for HARP clock in Open Ephys events (passed as harp_line).
    sniffing_folder
        Optional path to the Sniffing folder. If provided, adds a TimeSeries acquisition
        "sniff_detector" assembled from SniffDetector__32.bin across subfolders.
    data_folder, results_folder
        Base folders (defaults match your notebook).
    output_parent_tag
        Used in output folder name: {ephys_base_name}_{output_parent_tag}_{timestamp}
    scratch_dir
        Where behavior NWB is written and where harp salvage writes timestamps.
    experiment, recording
        Used to locate the source ephys NWB name.
    copy_metadata_jsons
        If True, copies {data_description, procedures, rig, session, subject}.json
        from the raw ephys base folder into the output folder.

    Returns
    -------
    Path
        Path to the merged output NWB (file or Zarr dir, depending on source).
    """
    data_folder = Path(data_folder)
    results_folder = Path(results_folder)
    scratch_dir = Path(scratch_dir)

    beh_json = Path(beh_json)
    ibl_app_output = Path(ibl_app_output)

    # --- derive names/paths exactly like the notebook ---
    ephys_base_name = ephys_sorted_name.split("_sorted")[0]
    ephys_raw_folder = data_folder / ephys_base_name

    nwb_loc = data_folder / ephys_sorted_name / "nwb" / f"{ephys_base_name}_experiment{experiment}_recording{recording}.nwb"

    now = datetime.datetime.now(ZoneInfo("America/Los_Angeles"))
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    new_nwb_folder = f"{ephys_base_name}_{output_parent_tag}_{date_time}"

    nwb_output_path = results_folder / new_nwb_folder / nwb_loc.name
    nwb_output_path.parent.mkdir(exist_ok=True, parents=True)

    # --- 1) bonsai JSON -> behavior NWB in scratch ---
    try:
        from foraging_gui.TransferToNWB import bonsai_to_nwb
    except Exception as e:
        raise ImportError(
            "Could not import foraging_gui.TransferToNWB.bonsai_to_nwb. "
            "Install/enable foraging_gui in this environment."
        ) from e

    bonsai_to_nwb(str(beh_json), str(scratch_dir))
    new_beh_path = scratch_dir / f"{beh_json.stem}.nwb"

    if not new_beh_path.exists():
        raise FileNotFoundError(f"bonsai_to_nwb did not produce expected file: {new_beh_path}")

    # --- 2) copy metadata JSONs (optional, notebook behavior) ---
    if copy_metadata_jsons:
        metadata_to_copy = ["data_description", "procedures", "rig", "session", "subject"]
        for stem in metadata_to_copy:
            src = ephys_raw_folder / f"{stem}.json"
            dst = nwb_output_path.parent / f"{stem}.json"
            if src.exists():
                shutil.copy(src, dst)

    # --- 3) merge behavior into ephys copy ---
    merge_behavior_into_ephys(
        ephys_path=str(nwb_loc),
        behavior_path=str(new_beh_path),
        out_path=str(nwb_output_path),
    )

    # --- 4) electrodes/units locations from IBL app output ---
    merged_elec = update_electrodes_table_locations_from_ibl_app(str(nwb_output_path), str(ibl_app_output))
    replace_electrodes_table_with_merged(str(nwb_output_path), merged_elec)

    # add {location, ccf_ml, ccf_ap, ccf_dv} columns into units
    add_units_locations_from_electrodes_zarr(nwb_zarr_path=str(nwb_output_path))

    # --- 5) optional HARP alignment + spike_times rewrite ---
    if do_harp_alignment:
        directory = str(data_folder / ephys_base_name / "ecephys" / "ecephys_clipped")
        harp_timestamp_salvage(
            directory,
            harp_line=int(harp_channel),
            output_parent=str(scratch_dir),
        )
        new_units_table = update_units_table_timestamps(
            str(nwb_output_path),
            ephys_sorted_name,
            temp_harp_dir=str(scratch_dir),
            experiment=f"experiment{experiment}",
            recording=f"recording{recording}",
            data_dir=str(data_folder),
        )
        write_new_unit_timestamps_to_file(str(nwb_output_path), new_units_table)

    # --- 6) optional sniffing acquisition ---
    if sniffing_folder is not None:
        sniffing_folder = Path(sniffing_folder)

        # import harp lazily so environments without it still work if sniffing_folder not used
        import harp

        sniff_events = []
        for loc in os.listdir(sniffing_folder):
            try:
                sniff_events.append(harp.read(str(sniffing_folder / loc / "SniffDetector__32.bin")))
            except FileNotFoundError:
                continue

        if len(sniff_events) == 0:
            raise FileNotFoundError(
                f"No SniffDetector__32.bin files found under sniffing_folder={sniffing_folder}"
            )

        sniff_events = pd.concat(sniff_events)
        sniff_ts = TimeSeries(
            name="sniff_detector",
            description="Sniff detector continuous signal imported from HARP SniffDetector__32.bin",
            data=np.asarray(sniff_events.values).reshape(len(sniff_events.values)),
            unit="a.u.",
            timestamps=np.asarray(sniff_events.index.values),
        )

        with NWBZarrIO(path=str(nwb_output_path), mode="r+") as io:
            nwb = io.read()

            # avoid collisions if rerun
            if sniff_ts.name in nwb.acquisition:
                i = 2
                base = "sniff_detector"
                while f"{base}_{i}" in nwb.acquisition:
                    i += 1
                sniff_ts.name = f"{base}_{i}"

            nwb.add_acquisition(sniff_ts)
            io.write(nwb)

    return nwb_output_path