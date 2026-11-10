# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Independent edge-coordinate reference shared by weighted correctness tests."""

import numpy as np


def reference(source, target, weights, events, cols):
    """Accumulate only active edges, without relying on tile metadata or perm."""
    weights = np.broadcast_to(np.asarray(weights, np.float64), source.shape)
    result = np.zeros((len(events), cols), np.float64)
    with np.errstate(invalid="ignore"):
        for b, event in enumerate(np.asarray(events, np.float64)):
            active = event[source] != 0
            np.add.at(result[b], target[active], event[source[active]] * weights[active])
    return result


def matrix(rows=269, cols=24581):
    """Unsorted CSR with duplicates, empty rows, long rows and four scatter stages."""
    lengths = np.resize([0, 1, 2, 7, 33, 65, 513], rows)
    source = np.repeat(np.arange(rows), lengths)
    target = (np.arange(source.size) * 37 + source * 13) % cols
    target[:8] = np.array([0, 8191, 8192, 12287, 12288, 24575, 24576, 24576]) % cols
    ptr = np.r_[0, lengths.cumsum()]
    return source, target, ptr
