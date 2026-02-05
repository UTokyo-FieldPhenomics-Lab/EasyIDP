# Optimize Logging Configuration

## Context
The user reported issues with repetitive logs drowning out useful information and conflicting with `tqdm` progress bars. Specifically:
1. "Converted to affine mode" appearing repeatedly (spamming).
2. Log output breaking the progress bar layout (interleaving).
3. Similar logs (e.g., "Reprojecting ROI") appearing too frequently.

## Changes
### `src/easyidp/__init__.py`
- Added imports for `time` and `tqdm`.
- Implemented `LogFilter` class to handle log deduplication and throttling.
  - **Deduplication**: Strictly blocks consecutive identical messages to prevent simple spam.
  - **Throttling**: Limits the frequency of specific message patterns to reduce noise.
    - Patterns configured: `"Converted to affine mode"`, `"Reprojecting ROI"`, `"GeoTiff successfully saved"`.
    - Default cooldown: 1.0 second.
- Implemented `tqdm_sink` to redirect log output to `tqdm.write`.
  - Ensures logs are printed *above* the active progress bar instead of overwriting/interleaving with it.
- Replaced the default `loguru` stderr handler with the new `tqdm_sink` and `LogFilter`.

## Verification
- Verified with a manual test script ensuring expected behavior:
  - Consecutive duplicate messages are blocked.
  - Frequent "similar" messages (matching throttle patterns) are sampled (shown once, then blocked for cooldown).
  - Messages from different valid groups or unique messages pass through correctly.
