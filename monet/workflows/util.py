# Author: Florian Wagner <florian.compbio@gmail.com>
# Copyright (c) 2021 Florian Wagner
#
# This file is part of Monet.

from ..visualize.util import DEFAULT_PLOTLY_COLORS, DEFAULT_GGPLOT_COLORS


def get_default_cluster_colors():
    cluster_colors = {}

    for i in range(20):
        # set default colors for first 20 clusters,
        # assuming they are named correctly
        # ("Cluster 1", "Cluster 2", ..., or "1", "2", ...)
        for i in range(20):
            # for 0, 1, ..., we use ggplot colors
            cluster_colors[i] = DEFAULT_GGPLOT_COLORS[i]
            # Cluster 1, Cluster 2, ..., we use plotly colors
            cluster_colors['Cluster %d' % (i+1)] = DEFAULT_PLOTLY_COLORS[i]

    # also set default color for cluster named "Outliers"
    cluster_colors['Outliers'] = 'lightgray'
    return cluster_colors
