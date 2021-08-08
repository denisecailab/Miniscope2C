import functools as fct
import os
import re
import shutil
from os import listdir
from os.path import isdir, isfile
from os.path import join as pjoin
from pathlib import Path
from typing import Callable, List, Optional, Union
from uuid import uuid4

import _operator
import cv2
import dask as da
import dask.array as darr
import ffmpeg
import numpy as np
import pandas as pd
import rechunker
import SimpleITK as sitk
import xarray as xr
import zarr as zr
from dask.core import flatten
from dask.delayed import optimize as default_delay_optimize
from dask.optimization import cull, fuse, inline, inline_functions
from dask.utils import ensure_dict
from distributed.diagnostics.plugin import SchedulerPlugin
from distributed.scheduler import SchedulerState, cast
from natsort import natsorted

ANNOTATIONS = {
    "from-zarr-store": {"resources": {"MEM": 1}},
    "load_avi_ffmpeg": {"resources": {"MEM": 1}},
    "est_motion_chunk": {"resources": {"MEM": 1}},
    "transform_perframe": {"resources": {"MEM": 0.5}},
    "pnr_perseed": {"resources": {"MEM": 0.5}},
    "ks_perseed": {"resources": {"MEM": 0.5}},
    "smooth_corr": {"resources": {"MEM": 1}},
    "vectorize_noise_fft": {"resources": {"MEM": 1}},
    "vectorize_noise_welch": {"resources": {"MEM": 1}},
    "update_spatial_block": {"resources": {"MEM": 1}},
    "tensordot_restricted": {"resources": {"MEM": 1}},
    "update_temporal_block": {"resources": {"MEM": 1}},
    "merge_restricted": {"resources": {"MEM": 1}},
}
"""
Dask annotations that should be applied to each task.
This is a `dict` mapping task names (actually patterns) to a `dict` of dask
annotations that should be applied to the tasks. It is mainly used to constrain
number of tasks that can be concurrently in memory for each worker.
See Also
-------
:doc:`distributed:resources`
"""

FAST_FUNCTIONS = [
    darr.core.getter_inline,
    darr.core.getter,
    _operator.getitem,
    zr.core.Array,
    darr.chunk.astype,
    darr.core.concatenate_axes,
    darr.core._vindex_slice,
    darr.core._vindex_merge,
    darr.core._vindex_transpose,
]
"""
List of fast functions that should be inlined during optimization.
See Also
-------
:doc:`dask:optimize`
"""


def map_ts(ts: pd.DataFrame) -> pd.DataFrame:
    """map frames from Cam1 to Cam0 with nearest neighbour using the timestamp
    file from miniscope recordings.

    Parameters
    ----------
    ts : pd.DataFrame
        input timestamp dataframe. should contain field 'frameNum', 'camNum' and
        'sysClock'

    Returns
    -------
    pd.DataFrame
        output dataframe. should contain field 'fmCam0' and 'fmCam1'
    """
    ts_sort = ts.sort_values("sysClock")
    ts_sort["ts_behav"] = np.where(ts_sort["camNum"] == 1, ts_sort["sysClock"], np.nan)
    ts_sort["ts_forward"] = ts_sort["ts_behav"].fillna(method="ffill")
    ts_sort["ts_backward"] = ts_sort["ts_behav"].fillna(method="bfill")
    ts_sort["diff_forward"] = np.absolute(ts_sort["sysClock"] - ts_sort["ts_forward"])
    ts_sort["diff_backward"] = np.absolute(ts_sort["sysClock"] - ts_sort["ts_backward"])
    ts_sort["fm_behav"] = np.where(ts_sort["camNum"] == 1, ts_sort["frameNum"], np.nan)
    ts_sort["fm_forward"] = ts_sort["fm_behav"].fillna(method="ffill")
    ts_sort["fm_backward"] = ts_sort["fm_behav"].fillna(method="bfill")
    ts_sort["fmCam1"] = np.where(
        ts_sort["diff_forward"] < ts_sort["diff_backward"],
        ts_sort["fm_forward"],
        ts_sort["fm_backward"],
    )
    ts_map = (
        ts_sort[ts_sort["camNum"] == 0][["frameNum", "fmCam1"]]
        .dropna()
        .rename(columns=dict(frameNum="fmCam0"))
        .astype(dict(fmCam1=int))
    )
    return ts_map


def norm(x: np.ndarray):
    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    diff = xmax - xmin
    if diff > 0:
        return (x - xmin) / diff
    else:
        return x


def norm_xr(x: xr.DataArray):
    xmin = x.min().compute().values
    xmax = x.max().compute().values
    diff = xmax - xmin
    if diff > 0:
        return (x - xmin) / diff
    else:
        return x


def open_minian(
    dpath: str, post_process: Optional[Callable] = None, return_dict=False
) -> Union[dict, xr.Dataset]:
    """
    Load an existing minian dataset.
    If `dpath` is a file, then it is assumed that the full dataset is saved as a
    single file, and this function will directly call
    :func:`xarray.open_dataset` on `dpath`. Otherwise if `dpath` is a directory,
    then it is assumed that the dataset is saved as a directory of `zarr`
    arrays, as produced by :func:`save_minian`. This function will then iterate
    through all the directories under input `dpath` and load them as
    `xr.DataArray` with `zarr` backend, so it is important that the user make
    sure every directory under `dpath` can be load this way. The loaded arrays
    will be combined as either a `xr.Dataset` or a `dict`. Optionally a
    user-supplied custom function can be used to post process the resulting
    `xr.Dataset`.
    Parameters
    ----------
    dpath : str
        The path to the minian dataset that should be loaded.
    post_process : Callable, optional
        User-supplied function to post process the dataset. Only used if
        `return_dict` is `False`. Two arguments will be passed to the function:
        the resulting dataset `ds` and the data path `dpath`. In other words the
        function should have signature `f(ds: xr.Dataset, dpath: str) ->
        xr.Dataset`. By default `None`.
    return_dict : bool, optional
        Whether to combine the DataArray as dictionary, where the `.name`
        attribute will be used as key. Otherwise the DataArray will be combined
        using `xr.merge(..., compat="no_conflicts")`, which will implicitly
        align the DataArray over all dimensions, so it is important to make sure
        the coordinates are compatible and will not result in creation of large
        NaN-padded results. Only used if `dpath` is a directory, otherwise a
        `xr.Dataset` is always returned. By default `False`.
    Returns
    -------
    ds : Union[dict, xr.Dataset]
        The resulting dataset. If `return_dict` is `True` it will be a `dict`,
        otherwise a `xr.Dataset`.
    See Also
    -------
    xarray.open_zarr : for how each directory will be loaded as `xr.DataArray`
    xarray.merge : for how the `xr.DataArray` will be merged as `xr.Dataset`
    """
    if isfile(dpath):
        ds = xr.open_dataset(dpath).chunk()
    elif isdir(dpath):
        dslist = []
        for d in listdir(dpath):
            arr_path = pjoin(dpath, d)
            if isdir(arr_path):
                arr = list(xr.open_zarr(arr_path).values())[0]
                arr.data = darr.from_zarr(
                    os.path.join(arr_path, arr.name), inline_array=True
                )
                dslist.append(arr)
        if return_dict:
            ds = {d.name: d for d in dslist}
        else:
            ds = xr.merge(dslist, compat="no_conflicts")
    if (not return_dict) and post_process:
        ds = post_process(ds, dpath)
    return ds


def load_videos(
    vpath: str,
    pattern=r"msCam[0-9]+\.avi$",
    dtype: Union[str, type] = np.float64,
    downsample: Optional[dict] = None,
    downsample_strategy="subset",
    post_process: Optional[Callable] = None,
) -> xr.DataArray:
    """
    Load multiple videos in a folder and return a `xr.DataArray`.
    Load videos from the folder specified in `vpath` and according to the regex
    `pattern`, then concatenate them together and return a `xr.DataArray`
    representation of the concatenated videos. The videos are sorted by
    filenames with :func:`natsort.natsorted` before concatenation. Optionally
    the data can be downsampled, and the user can pass in a custom callable to
    post-process the result.
    Parameters
    ----------
    vpath : str
        The path containing the videos to load.
    pattern : regexp, optional
        The regexp matching the filenames of the videso. By default
        `r"msCam[0-9]+\.avi$"`, which can be interpreted as filenames starting
        with "msCam" followed by at least a number, and then followed by ".avi".
    dtype : Union[str, type], optional
        Datatype of the resulting DataArray, by default `np.float64`.
    downsample : dict, optional
        A dictionary mapping dimension names to an integer downsampling factor.
        The dimension names should be one of "height", "width" or "frame". By
        default `None`.
    downsample_strategy : str, optional
        How the downsampling should be done. Only used if `downsample` is not
        `None`. Either `"subset"` where data points are taken at an interval
        specified in `downsample`, or `"mean"` where mean will be taken over
        data within each interval. By default `"subset"`.
    post_process : Callable, optional
        An user-supplied custom function to post-process the resulting array.
        Four arguments will be passed to the function: the resulting DataArray
        `varr`, the input path `vpath`, the list of matched video filenames
        `vlist`, and the list of DataArray before concatenation `varr_list`. The
        function should output another valide DataArray. In other words, the
        function should have signature `f(varr: xr.DataArray, vpath: str, vlist:
        List[str], varr_list: List[xr.DataArray]) -> xr.DataArray`. By default
        `None`
    Returns
    -------
    varr : xr.DataArray
        The resulting array representation of the input movie. Should have
        dimensions ("frame", "height", "width").
    Raises
    ------
    FileNotFoundError
        if no files under `vpath` match the pattern `pattern`
    ValueError
        if the matched files does not have extension ".avi", ".mkv" or ".tif"
    NotImplementedError
        if `downsample_strategy` is not "subset" or "mean"
    """
    vpath = os.path.normpath(vpath)
    vlist = natsorted(
        [vpath + os.sep + v for v in os.listdir(vpath) if re.search(pattern, v)]
    )
    if not vlist:
        raise FileNotFoundError(
            "No data with pattern {}"
            " found in the specified folder {}".format(pattern, vpath)
        )
    print("loading {} videos in folder {}".format(len(vlist), vpath))

    file_extension = os.path.splitext(vlist[0])[1]
    if file_extension in (".avi", ".mkv"):
        movie_load_func = load_avi_lazy
    else:
        raise ValueError("Extension not supported.")

    varr_list = [movie_load_func(v) for v in vlist]
    varr = darr.concatenate(varr_list, axis=0)
    varr = xr.DataArray(
        varr,
        dims=["frame", "height", "width"],
        coords=dict(
            frame=np.arange(varr.shape[0]),
            height=np.arange(varr.shape[1]),
            width=np.arange(varr.shape[2]),
        ),
    )
    if dtype:
        varr = varr.astype(dtype)
    if downsample:
        if downsample_strategy == "mean":
            varr = varr.coarsen(**downsample, boundary="trim", coord_func="min").mean()
        elif downsample_strategy == "subset":
            varr = varr.isel(**{d: slice(None, None, w) for d, w in downsample.items()})
        else:
            raise NotImplementedError("unrecognized downsampling strategy")
    varr = varr.rename("fluorescence")
    if post_process:
        varr = post_process(varr, vpath, vlist, varr_list)
    arr_opt = fct.partial(custom_arr_optimize, keep_patterns=["^load_avi_ffmpeg"])
    with da.config.set(array_optimize=arr_opt):
        varr = da.optimize(varr)[0]
    return varr


def load_avi_lazy(fname: str) -> darr.array:
    """
    Lazy load an avi video.
    This function construct a single delayed task for loading the video as a
    whole.
    Parameters
    ----------
    fname : str
        The filename of the video to load.
    Returns
    -------
    arr : darr.array
        The array representation of the video.
    """
    probe = ffmpeg.probe(fname)
    video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
    w = int(video_info["width"])
    h = int(video_info["height"])
    f = int(video_info["nb_frames"])
    return da.array.from_delayed(
        da.delayed(load_avi_ffmpeg)(fname, h, w, f), dtype=np.uint8, shape=(f, h, w)
    )


def load_avi_ffmpeg(fname: str, h: int, w: int, f: int) -> np.ndarray:
    """
    Load an avi video using `ffmpeg`.
    This function directly invoke `ffmpeg` using the `python-ffmpeg` wrapper and
    retrieve the data from buffer.
    Parameters
    ----------
    fname : str
        The filename of the video to load.
    h : int
        The height of the video.
    w : int
        The width of the video.
    f : int
        The number of frames in the video.
    Returns
    -------
    arr : np.ndarray
        The resulting array. Has shape (`f`, `h`, `w`).
    """
    out_bytes, err = (
        ffmpeg.input(fname)
        .video.output("pipe:", format="rawvideo", pix_fmt="gray")
        .run(capture_stdout=True)
    )
    return np.frombuffer(out_bytes, np.uint8).reshape(f, h, w)


def custom_arr_optimize(
    dsk: dict,
    keys: list,
    fast_funcs: list = FAST_FUNCTIONS,
    inline_patterns=[],
    rename_dict: Optional[dict] = None,
    rewrite_dict: Optional[dict] = None,
    keep_patterns=[],
) -> dict:
    """
    Customized implementation of array optimization function.
    Parameters
    ----------
    dsk : dict
        Input dask task graph.
    keys : list
        Output task keys.
    fast_funcs : list, optional
        List of fast functions to be inlined. By default :const:`FAST_FUNCTIONS`.
    inline_patterns : list, optional
        List of patterns of task keys to be inlined. By default `[]`.
    rename_dict : dict, optional
        Dictionary mapping old task keys to new ones. Only used during fusing of
        tasks. By default `None`.
    rewrite_dict : dict, optional
        Dictionary mapping old task key substrings to new ones. Applied at the
        end of optimization to all task keys. By default `None`.
    keep_patterns : list, optional
        List of patterns of task keys that should be preserved during
        optimization. By default `[]`.
    Returns
    -------
    dsk : dict
        Optimized dask graph.
    See Also
    -------
    :doc:`dask:optimize`
    `dask.array.optimization.optimize`
    """
    # inlining lots of array operations ref:
    # https://github.com/dask/dask/issues/6668
    if rename_dict:
        key_renamer = fct.partial(custom_fused_keys_renamer, rename_dict=rename_dict)
    else:
        key_renamer = custom_fused_keys_renamer
    keep_keys = []
    if keep_patterns:
        key_ls = list(dsk.keys())
        for pat in keep_patterns:
            keep_keys.extend(list(filter(lambda k: check_key(k, pat), key_ls)))
    dsk = darr.optimization.optimize(
        dsk,
        keys,
        fuse_keys=keep_keys,
        fast_functions=fast_funcs,
        rename_fused_keys=key_renamer,
    )
    if inline_patterns:
        dsk = inline_pattern(dsk, inline_patterns, inline_constants=False)
    if rewrite_dict:
        dsk_old = dsk.copy()
        for key, val in dsk_old.items():
            key_new = rewrite_key(key, rewrite_dict)
            if key_new != key:
                dsk[key_new] = val
                dsk[key] = key_new
    return dsk


def rewrite_key(key: Union[str, tuple], rwdict: dict) -> str:
    """
    Rewrite a task key according to `rwdict`.
    Parameters
    ----------
    key : Union[str, tuple]
        Input task key.
    rwdict : dict
        Dictionary mapping old task key substring to new ones. All keys in this
        dictionary that exists in input `key` will be substituted.
    Returns
    -------
    key : str
        The new key.
    Raises
    ------
    ValueError
        if input `key` is neither `str` or `tuple`
    """
    typ = type(key)
    if typ is tuple:
        k = key[0]
    elif typ is str:
        k = key
    else:
        raise ValueError("key must be either str or tuple: {}".format(key))
    for pat, repl in rwdict.items():
        k = re.sub(pat, repl, k)
    if typ is tuple:
        ret_key = list(key)
        ret_key[0] = k
        return tuple(ret_key)
    else:
        return k


def custom_fused_keys_renamer(
    keys: list, max_fused_key_length=120, rename_dict: Optional[dict] = None
) -> str:
    """
    Custom implmentation to create new keys for `fuse` tasks.
    Uses custom `split_key` implementation.
    Parameters
    ----------
    keys : list
        List of task keys that should be fused together.
    max_fused_key_length : int, optional
        Used to limit the maximum string length for each renamed key. If `None`,
        there is no limit. By default `120`.
    rename_dict : dict, optional
        Dictionary used to rename keys during fuse. By default `None`.
    Returns
    -------
    fused_key : str
        The fused task key.
    See Also
    -------
    split_key
    dask.optimization.fuse
    """
    it = reversed(keys)
    first_key = next(it)
    typ = type(first_key)

    if max_fused_key_length:  # Take into account size of hash suffix
        max_fused_key_length -= 5

    def _enforce_max_key_limit(key_name):
        if max_fused_key_length and len(key_name) > max_fused_key_length:
            name_hash = f"{hash(key_name):x}"[:4]
            key_name = f"{key_name[:max_fused_key_length]}-{name_hash}"
        return key_name

    if typ is str:
        first_name = split_key(first_key, rename_dict=rename_dict)
        names = {split_key(k, rename_dict=rename_dict) for k in it}
        names.discard(first_name)
        names = sorted(names)
        names.append(first_key)
        concatenated_name = "-".join(names)
        return _enforce_max_key_limit(concatenated_name)
    elif typ is tuple and len(first_key) > 0 and isinstance(first_key[0], str):
        first_name = split_key(first_key, rename_dict=rename_dict)
        names = {split_key(k, rename_dict=rename_dict) for k in it}
        names.discard(first_name)
        names = sorted(names)
        names.append(first_key[0])
        concatenated_name = "-".join(names)
        return (_enforce_max_key_limit(concatenated_name),) + first_key[1:]


def split_key(key: Union[tuple, str], rename_dict: Optional[dict] = None) -> str:
    """
    Split, rename and filter task keys.
    This is custom implementation that only keeps keys found in :const:`ANNOTATIONS`.
    Parameters
    ----------
    key : Union[tuple, str]
        The input task key.
    rename_dict : dict, optional
        Dictionary used to rename keys. By default `None`.
    Returns
    -------
    new_key : str
        New key.
    """
    if type(key) is tuple:
        key = key[0]
    kls = key.split("-")
    if rename_dict:
        kls = list(map(lambda k: rename_dict.get(k, k), kls))
    kls_ft = list(filter(lambda k: k in ANNOTATIONS.keys(), kls))
    if kls_ft:
        return "-".join(kls_ft)
    else:
        return kls[0]


def check_key(key: Union[str, tuple], pat: str) -> bool:
    """
    Check whether `key` contains pattern.
    Parameters
    ----------
    key : Union[str, tuple]
        Input key. If a `tuple` then the first element will be used to check.
    pat : str
        Pattern to check.
    Returns
    -------
    bool
        Whether `key` contains pattern.
    """
    try:
        return bool(re.search(pat, key))
    except TypeError:
        return bool(re.search(pat, key[0]))


def check_pat(key: Union[str, tuple], pat_ls: List[str]) -> bool:
    """
    Check whether `key` contains any pattern in a list.
    Parameters
    ----------
    key : Union[str, tuple]
        Input key. If a `tuple` then the first element will be used to check.
    pat_ls : List[str]
        List of pattern to check.
    Returns
    -------
    bool
        Whether `key` contains any pattern in the list.
    """
    for pat in pat_ls:
        if check_key(key, pat):
            return True
    return False


def inline_pattern(dsk: dict, pat_ls: List[str], inline_constants: bool) -> dict:
    """
    Inline tasks whose keys match certain patterns.
    Parameters
    ----------
    dsk : dict
        Input dask graph.
    pat_ls : List[str]
        List of patterns to check.
    inline_constants : bool
        Whether to inline constants.
    Returns
    -------
    dsk : dict
        Dask graph with keys inlined.
    See Also
    -------
    dask.optimization.inline
    """
    keys = [k for k in dsk.keys() if check_pat(k, pat_ls)]
    if keys:
        dsk = inline(dsk, keys, inline_constants=inline_constants)
        for k in keys:
            del dsk[k]
        if inline_constants:
            dsk, dep = cull(dsk, set(list(flatten(keys))))
    return dsk


def custom_delay_optimize(
    dsk: dict, keys: list, fast_functions=[], inline_patterns=[], **kwargs
) -> dict:
    """
    Custom optimization functions for delayed tasks.
    By default only fusing of tasks will be carried out.
    Parameters
    ----------
    dsk : dict
        Input dask task graph.
    keys : list
        Output task keys.
    fast_functions : list, optional
        List of fast functions to be inlined. By default `[]`.
    inline_patterns : list, optional
        List of patterns of task keys to be inlined. By default `[]`.
    Returns
    -------
    dsk : dict
        Optimized dask graph.
    """
    dsk, _ = fuse(ensure_dict(dsk), rename_keys=custom_fused_keys_renamer)
    if inline_patterns:
        dsk = inline_pattern(dsk, inline_patterns, inline_constants=False)
    if fast_functions:
        dsk = inline_functions(
            dsk,
            [],
            fast_functions=fast_functions,
        )
    return dsk


def unique_keys(keys: list) -> np.ndarray:
    """
    Returns only unique keys in a list of task keys.
    Dask task keys regarding arrays are usually tuples representing chunked
    operations. This function ignore different chunks and only return unique keys.
    Parameters
    ----------
    keys : list
        List of dask keys.
    Returns
    -------
    unique : np.ndarray
        Unique keys.
    """
    new_keys = []
    for k in keys:
        if isinstance(k, tuple):
            new_keys.append("chunked-" + k[0])
        elif isinstance(k, str):
            new_keys.append(k)
    return np.unique(new_keys)


def get_keys_pat(pat: str, keys: list, return_all=False) -> Union[list, str]:
    """
    Filter a list of task keys by pattern.
    Parameters
    ----------
    pat : str
        Pattern to check.
    keys : list
        List of keys to be filtered.
    return_all : bool, optional
        Whether to return all keys matching `pat`. If `False` then only the
        first match will be returned. By default `False`.
    Returns
    -------
    keys : Union[list, str]
        If `return_all` is `True` then a list of keys will be returned.
        Otherwise only one key will be returned.
    """
    keys_filt = list(filter(lambda k: check_key(k, pat), list(keys)))
    if return_all:
        return keys_filt
    else:
        return keys_filt[0]


def save_minian(
    var: xr.DataArray,
    dpath: str,
    meta_dict: Optional[dict] = None,
    overwrite=False,
    chunks: Optional[dict] = None,
    compute=True,
    mem_limit="500MB",
) -> xr.DataArray:
    """
    Save a `xr.DataArray` with `zarr` storage backend following minian
    conventions.
    This function will store arbitrary `xr.DataArray` into `dpath` with `zarr`
    backend. A separate folder will be created under `dpath`, with folder name
    `var.name + ".zarr"`. Optionally metadata can be retrieved from directory
    hierarchy and added as coordinates of the `xr.DataArray`. In addition, an
    on-disk rechunking of the result can be performed using
    :func:`rechunker.rechunk` if `chunks` are given.
    Parameters
    ----------
    var : xr.DataArray
        The array to be saved.
    dpath : str
        The path to the minian dataset directory.
    meta_dict : dict, optional
        How metadata should be retrieved from directory hierarchy. The keys
        should be negative integers representing directory level relative to
        `dpath` (so `-1` means the immediate parent directory of `dpath`), and
        values should be the name of dimensions represented by the corresponding
        level of directory. The actual coordinate value of the dimensions will
        be the directory name of corresponding level. By default `None`.
    overwrite : bool, optional
        Whether to overwrite the result on disk. By default `False`.
    chunks : dict, optional
        A dictionary specifying the desired chunk size. The chunk size should be
        specified using :doc:`dask:array-chunks` convention, except the "auto"
        specifiication is not supported. The rechunking operation will be
        carried out with on-disk algorithms using :func:`rechunker.rechunk`. By
        default `None`.
    compute : bool, optional
        Whether to compute `var` and save it immediately. By default `True`.
    mem_limit : str, optional
        The memory limit for the on-disk rechunking algorithm, passed to
        :func:`rechunker.rechunk`. Only used if `chunks` is not `None`. By
        default `"500MB"`.
    Returns
    -------
    var : xr.DataArray
        The array representation of saving result. If `compute` is `True`, then
        the returned array will only contain delayed task of loading the on-disk
        `zarr` arrays. Otherwise all computation leading to the input `var` will
        be preserved in the result.
    Examples
    -------
    The following will save the variable `var` to directory
    `/spatial_memory/alpha/learning1/minian/important_array.zarr`, with the
    additional coordinates: `{"session": "learning1", "animal": "alpha",
    "experiment": "spatial_memory"}`.
    >>> save_minian(
    ...     var.rename("important_array"),
    ...     "/spatial_memory/alpha/learning1/minian",
    ...     {-1: "session", -2: "animal", -3: "experiment"},
    ... ) # doctest: +SKIP
    """
    dpath = os.path.normpath(dpath)
    Path(dpath).mkdir(parents=True, exist_ok=True)
    ds = var.to_dataset()
    if meta_dict is not None:
        pathlist = os.path.split(os.path.abspath(dpath))[0].split(os.sep)
        ds = ds.assign_coords(
            **dict([(dn, pathlist[di]) for dn, di in meta_dict.items()])
        )
    md = {True: "a", False: "w-"}[overwrite]
    fp = os.path.join(dpath, var.name + ".zarr")
    if overwrite:
        try:
            shutil.rmtree(fp)
        except FileNotFoundError:
            pass
    arr = ds.to_zarr(fp, compute=compute, mode=md)
    if (chunks is not None) and compute:
        chunks = {d: var.sizes[d] if v <= 0 else v for d, v in chunks.items()}
        dst_path = os.path.join(dpath, str(uuid4()))
        temp_path = os.path.join(dpath, str(uuid4()))
        with da.config.set(
            array_optimize=darr.optimization.optimize,
            delayed_optimize=default_delay_optimize,
        ):
            zstore = zr.open(fp)
            rechk = rechunker.rechunk(
                zstore[var.name], chunks, mem_limit, dst_path, temp_store=temp_path
            )
            rechk.execute()
        try:
            shutil.rmtree(temp_path)
        except FileNotFoundError:
            pass
        arr_path = os.path.join(fp, var.name)
        for f in os.listdir(arr_path):
            os.remove(os.path.join(arr_path, f))
        for f in os.listdir(dst_path):
            os.rename(os.path.join(dst_path, f), os.path.join(arr_path, f))
        os.rmdir(dst_path)
    if compute:
        arr = xr.open_zarr(fp)[var.name]
        arr.data = darr.from_zarr(os.path.join(fp, var.name), inline_array=True)
    return arr


def xrconcat_recursive(var: Union[dict, list], dims: List[str]) -> xr.Dataset:
    """
    Recursively concatenate `xr.DataArray` over multiple dimensions.
    Parameters
    ----------
    var : Union[dict, list]
        Either a `dict` or a `list` of `xr.DataArray` to be concatenated. If a
        `dict` then keys should be `tuple`, with length same as the length of
        `dims` and values corresponding to the coordinates that uniquely
        identify each `xr.DataArray`. If a `list` then each `xr.DataArray`
        should contain valid coordinates for each dimensions specified in
        `dims`.
    dims : List[str]
        Dimensions to be concatenated over.
    Returns
    -------
    ds : xr.Dataset
        The concatenated dataset.
    Raises
    ------
    NotImplementedError
        if input `var` is neither a `dict` nor a `list`
    """
    if len(dims) > 1:
        if type(var) is dict:
            var_dict = var
        elif type(var) is list:
            var_dict = {tuple([np.asscalar(v[d]) for d in dims]): v for v in var}
        else:
            raise NotImplementedError("type {} not supported".format(type(var)))
        try:
            var_dict = {k: v.to_dataset() for k, v in var_dict.items()}
        except AttributeError:
            pass
        var_ps = pd.Series(var_dict)
        var_ps.index.set_names(dims, inplace=True)
        xr_ls = []
        for idx, v in var_ps.groupby(level=dims[0]):
            v.index = v.index.droplevel(dims[0])
            xarr = xrconcat_recursive(v.to_dict(), dims[1:])
            xr_ls.append(xarr)
        return xr.concat(xr_ls, dim=dims[0])
    else:
        if type(var) is dict:
            var = list(var.values())
        return xr.concat(var, dim=dims[0])


def load_v4_folder(dpath, folders=["miniscope_top", "miniscope_side"]):
    va0 = load_videos(
        os.path.join(dpath, folders[0]), pattern=r"[0-9]+\.avi$", dtype=np.uint8
    )
    va1 = load_videos(
        os.path.join(dpath, folders[1]), pattern=r"[0-9]+\.avi$", dtype=np.uint8
    )
    ts0 = pd.read_csv(os.path.join(dpath, folders[0], "timeStamps.csv"))
    ts1 = pd.read_csv(os.path.join(dpath, folders[1], "timeStamps.csv"))
    ts0["camNum"] = 0
    ts1["camNum"] = 1
    ts = (
        pd.concat([ts0, ts1], axis="index")
        .rename(
            {"Frame Number": "frameNum", "Time Stamp (ms)": "sysClock"}, axis="columns"
        )
        .sort_values("sysClock")
        .reset_index(drop=True)
    )
    ts_map = map_ts(ts).dropna().astype({"fmCam0": int, "fmCam1": int})
    va0 = va0.sel(frame=ts_map["fmCam0"].values).chunk({"frame": "auto"})
    va0 = va0.assign_coords(frame=np.arange(va0.sizes["frame"]))
    va1 = va1.sel(frame=ts_map["fmCam1"].values).chunk({"frame": "auto"})
    va1 = va1.assign_coords(frame=np.arange(va1.sizes["frame"]))
    return va0, va1


def write_video(
    arr: xr.DataArray,
    vname: Optional[str] = None,
    vpath: Optional[str] = ".",
    norm=True,
    options={"crf": "18", "preset": "ultrafast"},
) -> str:
    """
    Write a video from a movie array using `python-ffmpeg`.
    Parameters
    ----------
    arr : xr.DataArray
        Input movie array. Should have dimensions: ("frame", "height", "width")
        and should only be chunked along the "frame" dimension.
    vname : str, optional
        The name of output video. If `None` then a random one will be generated
        using :func:`uuid4.uuid`. By default `None`.
    vpath : str, optional
        The path to the folder containing the video. By default `"."`.
    norm : bool, optional
        Whether to normalize the values of the input array such that they span
        the full pixel depth range (0, 255). By default `True`.
    options : dict, optional
        Optional output arguments passed to `ffmpeg`. By default `{"crf": "18",
        "preset": "ultrafast"}`.
    Returns
    -------
    fname : str
        The absolute path to the video file.
    See Also
    --------
    ffmpeg.output
    """
    if not vname:
        vname = "{}.mp4".format(uuid4())
    fname = os.path.join(vpath, vname)
    if norm:
        arr_opt = fct.partial(
            custom_arr_optimize, rename_dict={"rechunk": "merge_restricted"}
        )
        with da.config.set(array_optimize=arr_opt):
            arr = arr.astype(np.float32)
            arr_max = arr.max().compute().values
            arr_min = arr.min().compute().values
        den = arr_max - arr_min
        arr -= arr_min
        arr /= den
        arr *= 255
    arr = arr.clip(0, 255).astype(np.uint8)
    w, h = arr.sizes["width"], arr.sizes["height"]
    process = (
        ffmpeg.input("pipe:", format="rawvideo", pix_fmt="gray", s="{}x{}".format(w, h))
        .filter("pad", int(np.ceil(w / 2) * 2), int(np.ceil(h / 2) * 2))
        .output(fname, pix_fmt="yuv420p", vcodec="libx264", r=30, **options)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    for blk in arr.data.blocks:
        process.stdin.write(np.array(blk).tobytes())
    process.stdin.close()
    process.wait()
    return fname
