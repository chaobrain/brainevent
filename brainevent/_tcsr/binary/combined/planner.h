// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
// Licensed under the Apache License, Version 2.0:
// http://www.apache.org/licenses/LICENSE-2.0

#ifndef COMBINED_PLANNER_H
#define COMBINED_PLANNER_H

#include <cstdint>

namespace combined
{
    constexpr int kTargetBlocks = 188;
    constexpr int kTileSharedBytes = 96 * 1024;
    constexpr int kWidthAlignment = 32;

    // Pure host planning, shared by metadata preparation and CUDA launch. The
    // metadata stride uses tile_count + 1, never the padded grid_tiles + 1.
    struct TileSchedule
    {
        int tile_width = 0;
        int tile_count = 0;
        int grid_tiles = 0;
        int block_size = 0;
        int shared_bytes = 0;
        int64_t scatter_blocks = 0;
        int64_t useful_blocks = 0;
    };

    constexpr TileSchedule make_tile_schedule(int cols, int batch, int accumulator_bytes = 4)
    {
        if (cols < 0 || batch < 0 || batch > 65535 ||
            (accumulator_bytes != 4 && accumulator_bytes != 8))
            return {};
        int width = kTileSharedBytes / accumulator_bytes;
        int required_tiles = 0;
        // Preserve valid zero-size plans without division by zero. No scatter is
        // launched, but metadata preparation can still use a positive width.
        if (cols > 0 && batch > 0)
        {
            required_tiles = (kTargetBlocks + batch - 1) / batch;
            if (required_tiles > 1)
            {
                const int bound = (cols - 1) / (required_tiles - 1);
                if (bound < width)
                    width = bound;
            }
            if (width >= kWidthAlignment)
                width -= width % kWidthAlignment;
            if (width < 1)
                width = 1;
            if (width > cols)
                width = cols;
        }
        const int tiles = static_cast<int>((static_cast<int64_t>(cols) + width - 1) / width);
        const int grid_tiles = cols == 0 || batch == 0 ? 0 : (tiles < required_tiles ? required_tiles : tiles);
        const int threads = width <= 4096 ? 256 : (width <= 12288 ? 512 : 1024);
        return {width, tiles, grid_tiles, threads, width * accumulator_bytes,
                static_cast<int64_t>(grid_tiles) * batch, static_cast<int64_t>(tiles) * batch};
    }
} // namespace combined

#endif
