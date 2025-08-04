Manual rechunking process
==========================

With versions of zarr starting with 3.0, in datasets being appended to the main zarr, dask chunks that would span multiple zarr chunks
in the final dataset will cause the parse to fail. For example, suppose that your goal is to have chunks in the time dimension of length
10, and your initial dataset has chunks like `(10, 10, 10, 7)` for a total of 37 time steps. Then, if you attempt to append a new xarray
dataset of time length 14 chunked like `(10, 4)`. You will see an Exception like:
```
ValueError: Specified Zarr chunks encoding['chunks']=(10,) for variable named 'x' would overlap multiple Dask chunks. Check the chunk at position 0, which has a size of 10 on dimension 0. It is unaligned with backend chunks of size 10 in region slice(37, None, None). Writing this array in parallel with Dask could lead to corrupted data. To resolve this issue, consider one of the following options: - Rechunk the array using `chunk()`. - Modify or delete `encoding['chunks']`. - Set `safe_chunks=False`. - Enable automatic chunks alignment with `align_chunks=True
```
This means that the first dask chunk of the update dataset, once written to the final zarr, would span two zarr chunks, which is not automatically handled in newer zarr versions. You can see this by looking at the desired final zarr in the example above, which we want to look like `(10, 10, 10, 10, 10, 1)`. The update dataset's first dask chunk would be spread between the 4th and 5th zarr chunks, triggering the failure. In addition, none of the suggestions in the Exception above actually fix the issue.

The fix is to manually rechunk the update dataset to have a pattern of dask chunks that are neatly aligned with the initial zarr's chunks. In this case, we rechunk the update dataset to look like `(3, 10, 1)`. Now, when we append this to the initial zarr, the first dask chunk of length 3 aligns neatly into the first zarr chunk in the initial dataset of length 7 to produce output chunks of length 10, preventing the multiple-overlap issue.

(Insert reference to code doing this, once it's added to gridded-etl-tools)