# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


def visualize(
    results,
    title='Acceleration Ratio',
    filename=None
):
    labels = list(results.keys())
    ratio = list(results.values())

    x = np.arange(len(labels))  # x-axis positions
    width = 0.35  # bar width

    fig, ax = plt.subplots()
    bars = ax.bar(x, ratio, width, label='Ratio')

    # axis labels
    ax.set_xlabel('Configurations')
    ax.set_ylabel('Acceleration Ratio')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    # annotate each bar with its value
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f'{height:.2f}',  # formatted value
            xy=(bar.get_x() + bar.get_width() / 2, height),  # label anchor
            xytext=(0, 3),  # offset
            textcoords="offset points",
            ha='center',
            va='bottom'
        )

    fig.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    plt.show()
