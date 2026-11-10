# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Dependency-free tile scheduling shared by preparation and benchmark sizing.

Keep the integer rules in sync with ``planner.h``. The target is a minimum
number of launched scatter CTAs; it does not measure SM occupancy.
"""

from dataclasses import dataclass


TARGET_BLOCKS = 188
MAX_SHARED_BYTES = 96 * 1024
WIDTH_ALIGNMENT = 32
INT32_MAX = (1 << 31) - 1


@dataclass(frozen=True)
class TileSchedule:
    tile_width: int = 0
    tile_count: int = 0
    grid_tiles: int = 0
    block_size: int = 0
    shared_bytes: int = 0
    scatter_blocks: int = 0
    useful_blocks: int = 0


def tile_schedule(cols: int, batch: int, accumulator_bytes: int = 4) -> TileSchedule:
    """Choose matching metadata and launch geometry for a typed accumulator.

    Invalid dimensions return an all-zero schedule. Zero columns or batches
    retain the maximum metadata tile width and launch no scatter CTAs.
    ``tile_count`` excludes any padding CTAs in ``grid_tiles``.
    """
    if not (0 <= cols <= INT32_MAX and 0 <= batch <= 65535) or accumulator_bytes not in (4, 8):
        return TileSchedule()
    width = MAX_SHARED_BYTES // accumulator_bytes
    if cols > 0 and batch > 0:
        required_tiles = (TARGET_BLOCKS + batch - 1) // batch
        if required_tiles > 1:
            width = min(width, (cols - 1) // (required_tiles - 1))
        if width >= WIDTH_ALIGNMENT:
            width -= width % WIDTH_ALIGNMENT
        width = min(cols, max(1, width))
        tiles = (cols + width - 1) // width
        grid_tiles = max(tiles, required_tiles)
    else:
        tiles = (cols + width - 1) // width
        grid_tiles = 0
    threads = 256 if width <= 4096 else 512 if width <= 12288 else 1024
    return TileSchedule(width, tiles, grid_tiles, threads, width * accumulator_bytes,
                        grid_tiles * batch, tiles * batch)
