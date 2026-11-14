# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Validate the actual C++ and Python planners without CUDA or a GPU."""

from dataclasses import astuple
from pathlib import Path
import random
import shutil
import subprocess

import pytest

from brainevent._tcsr.binary.combined.planner import TileSchedule, tile_schedule


@pytest.mark.parametrize("accumulator_bytes", [4, 8])
def test_launch_floor_and_metadata_geometry(accumulator_bytes):
    for cols in [1, 2, 31, 32, 33, 187, 188, 189, 4096, 12288, 24576, 100000, 2000000, (1 << 31) - 1]:
        for batch in [1, 2, 3, 4, 8, 32, 94, 187, 188, 189, 65535]:
            plan = tile_schedule(cols, batch, accumulator_bytes)
            assert 1 <= plan.tile_width <= min(cols, 96 * 1024 // accumulator_bytes)
            assert plan.tile_width < 32 or plan.tile_width % 32 == 0 or plan.tile_width == cols
            assert (plan.tile_count - 1) * plan.tile_width < cols <= plan.tile_count * plan.tile_width
            assert plan.scatter_blocks == batch * plan.grid_tiles >= 188
            assert plan.useful_blocks == batch * plan.tile_count
            assert (plan.useful_blocks >= 188) == (batch * cols >= 188)
            assert plan.grid_tiles >= plan.tile_count
            assert plan.grid_tiles == plan.tile_count or batch * cols < 188
            assert plan.shared_bytes == plan.tile_width * accumulator_bytes <= 96 * 1024
            assert plan.block_size == (256 if plan.tile_width <= 4096 else 512 if plan.tile_width <= 12288 else 1024)


@pytest.mark.parametrize("cols,batch", [(0, 0), (0, 1), (0, 65535), (1, 0), (24577, 0)])
@pytest.mark.parametrize("accumulator_bytes", [4, 8])
def test_empty_dimensions_keep_metadata_width(cols, batch, accumulator_bytes):
    plan = tile_schedule(cols, batch, accumulator_bytes)
    assert plan.tile_width == 96 * 1024 // accumulator_bytes
    assert plan.tile_count == (cols + plan.tile_width - 1) // plan.tile_width
    assert plan.grid_tiles == plan.scatter_blocks == plan.useful_blocks == 0


@pytest.mark.parametrize("cols,batch,accumulator_bytes", [(-1, 2, 4), (2, -1, 4), (2, 65536, 4),
                                                       (2, 2, 0), (2, 2, 2), (2, 2, 16)])
def test_invalid_schedule(cols, batch, accumulator_bytes):
    assert tile_schedule(cols, batch, accumulator_bytes) == TileSchedule()


@pytest.mark.parametrize("cols,batch,width,blocks", [(100000, 2, 1056, 190), (100000, 4, 2144, 188),
                                                   (400000, 2, 4288, 188), (1200000, 4, 24576, 196)])
def test_known_small_batch_schedules(cols, batch, width, blocks):
    plan = tile_schedule(cols, batch)
    assert (plan.tile_width, plan.scatter_blocks) == (width, blocks)


@pytest.fixture(scope="module")
def cpp_planner(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("requires g++ for C++/Python planner parity")
    root = tmp_path_factory.mktemp("combined_planner")
    source, executable = root / "planner.cpp", root / "planner"
    header = Path(__file__).resolve().parents[1] / "planner.h"
    source.write_text(f'''#include "{header}"
#include <iostream>
int main() {{
  int cols, batch, bytes;
  while (std::cin >> cols >> batch >> bytes) {{
    const auto p = combined::make_tile_schedule(cols, batch, bytes);
    std::cout << p.tile_width << ' ' << p.tile_count << ' ' << p.grid_tiles << ' '
              << p.block_size << ' ' << p.shared_bytes << ' ' << p.scatter_blocks
              << ' ' << p.useful_blocks << '\\n';
  }}
}}
''')
    built = subprocess.run([compiler, "-std=c++17", "-O2", str(source), "-o", str(executable)],
                           capture_output=True, text=True, timeout=60)
    assert built.returncode == 0, built.stderr
    return executable


def test_cpp_and_python_schedules_match(cpp_planner):
    cols = [-1, 0, 1, 2, 31, 32, 33, 187, 188, 189, 4095, 4096, 4097, 12287, 12288,
            12289, 24575, 24576, 24577, 100000, 2000000, (1 << 31) - 1]
    batches = [-1, 0, 1, 2, 3, 4, 8, 94, 187, 188, 189, 65535, 65536]
    cases = [(n, b, a) for n in cols for b in batches for a in (2, 4, 8)]
    rng = random.Random(188)
    cases.extend((rng.randrange(1, (1 << 31)), rng.randrange(1, 65536), rng.choice((4, 8)))
                 for _ in range(1000))
    run = subprocess.run([str(cpp_planner)], input="".join(f"{n} {b} {a}\n" for n, b, a in cases),
                         capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    actual = [tuple(map(int, line.split())) for line in run.stdout.splitlines()]
    assert actual == [astuple(tile_schedule(*case)) for case in cases]
