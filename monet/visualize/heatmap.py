# Copyright (c) 2020, 2021 Florian Wagner

# This file is part of Monet.

from pkg_resources import resource_filename
from typing import List, Tuple
import copy

import plotly.graph_objs as go
import numpy as np

from .util import DEFAULT_PLOTLY_COLORS, DEFAULT_GGPLOT_COLORS, load_colorscale
from .plotly_subtype import PlotlySubtype


class HeatmapPanel(PlotlySubtype):

    _default_colorscale_file = resource_filename('monet', 'data/RdBu_r_colormap.tsv')

    _default_colorscale = load_colorscale(_default_colorscale_file)

    _subtype_props = {'matrix', 'height', 'title', 'yaxis', 'colorbarlabel', 'tickfont'}
    _parent_type = go.Heatmap
    _default_values = dict(
        xaxis='x',
        hoverinfo='x+y+z',
        colorscale=_default_colorscale,
        height=1.0,
        colorbar=dict(
            len=0.3,
            outlinewidth=0,
            outlinecolor='black',
            titleside='right',
            ticks='outside',
            ticklen=5,
            thickness=20,
            title=dict(side='right'),
        )
    )


class HeatmapAnnotation(PlotlySubtype):

    _subtype_props = {
        'labels', 'clusterorder', 'clustercolors', 'clusterlabels',
        'colorscheme', 'height', 'title', 'yaxis', 'colorbarlabel', 'tickfont'}
    _parent_type = go.Heatmap
    _default_values = dict(
        yaxis=dict(tickfont=dict(size=24)),
        showscale=False,
        title=None,
        colorscheme='plotly')


class HeatmapLayout(PlotlySubtype):

    _subtype_props = {'panelmargin'}
    _parent_type = go.Layout
    _default_values = dict(
        panelmargin=20,
        font=dict(family='serif', size=24),
        width=1000,
        height=1000,
        margin=dict(l=150, b=20, t=100),
        xaxis=dict(
            #linecolor='black',
            #linewidth=1,
            showline=True,
            zeroline=False,
            showticklabels=False))


class Heatmap:

    # underscores are not valid in property names

    _default_yaxis = dict(
        tickfont=dict(family='serif', size=12),
        titlefont=dict(family='serif', size=24),
        ticks='outside',
        ticklen=5,
        linecolor='black',
        linewidth=1,
        showline=True,
        zeroline=False,
    )

    def __init__(self, data, layout):
        self.data = data
        self.layout = layout


    def get_figure(self):

        data = []

        # initialize layout with default values
        layout = self.layout.get_parent_object()

        layout_height = layout.height
        margin_top = layout.margin.t
        margin_bottom = layout.margin.b
        panelmargin = self.layout['panelmargin']

        # calculate actual height of heatmap (in px)
        heatmap_height_px = layout_height - margin_top - margin_bottom

        # calculate height available for actual data
        # (in px, excluding panel margin space)
        total_panel_height_px = heatmap_height_px - panelmargin * (len(self.data) - 1)

        # check if margins take up entire heatmap height
        if total_panel_height_px <= 0:
            raise ValueError(
                'Margins take up entire heatmap height! '
                'Try a lower `panelmargin` value (currently: %d).' % panelmargin)

        panelmargin_frac = panelmargin / heatmap_height_px

        # calculate combined panel heights
        total_height = 0
        for panel in self.data:
            if isinstance(panel, HeatmapAnnotation):
                continue
            total_height += panel.data['height']

        # calculate fraction of height taken up by data
        # (as opposed to margins)
        data_height_frac = total_panel_height_px / heatmap_height_px

        cur_y = 1.0
        for heatmap_index, heatmap_data in enumerate(self.data):

            if isinstance(heatmap_data, HeatmapPanel):

                panel = heatmap_data

                trace = panel.get_parent_object()

                trace['x'] = panel.data['matrix'].cells
                trace['y'] = panel.data['matrix'].genes
                trace['z'] = panel.data['matrix'].values

                # set heatmap y-axis
                if heatmap_index == 0:
                    trace['yaxis'] = 'y'
                else:
                    trace['yaxis'] = 'y%d' % (heatmap_index + 1)
                trace['xaxis'] = 'x'

                # set colorbar
                colorbar = trace['colorbar']
                colorbar_label = panel.data.get('colorbarlabel', None)
                if colorbar_label is not None and colorbar['title']['text'] is None:
                    colorbar['title']['text'] = colorbar_label

                data.append(trace)

                # calculate relative height (for `domain` property)
                #height_frac = (panel.data['height'] / total_height) * data_height_frac
                height_frac = (panel.data['height'] / total_height) * data_height_frac

                ### configure y-axis
                title = panel.data.get('title', None)
                tickfont = panel.data.get('tickfont', {})

                if heatmap_index == 0:
                    layout_yaxis_key = 'yaxis'
                else:
                    layout_yaxis_key = 'yaxis%d' % (heatmap_index + 1)

                #axis = layout.pop(layout_yaxis_key, {})
                yaxis = go.layout.YAxis(self._default_yaxis)

                yaxis['domain'] = [max(cur_y - height_frac, 0), cur_y]
                yaxis['title'] = title
                yaxis['autorange'] = 'reversed'
                yaxis['tickfont'].update(tickfont)

                yaxis_data = panel.data.get('yaxis', None)
                if yaxis_data is None:
                    yaxis_data = {}
                yaxis.update(yaxis_data)

                layout[layout_yaxis_key] = yaxis
                cur_y -= (height_frac + panelmargin_frac)

            elif isinstance(heatmap_data, HeatmapAnnotation):

                ann = heatmap_data

                # determine order of all clusters present
                cluster_order = ann.data.get('clusterorder', None)
                vc = ann.data['labels'].value_counts()
                if cluster_order is None:
                    cluster_order = vc.index.tolist()
                else:
                    cluster_order = cluster_order.copy()
                    for cluster in vc.index:
                        if cluster not in cluster_order:
                            cluster_order.append(cluster)

                # determine cluster colors
                cluster_colors = ann.data.get('clustercolors', None)
                if cluster_colors is None:
                    cluster_colors = {}
                else:
                    cluster_colors = copy.deepcopy(cluster_colors)

                if ann.data.get('colorscheme', None) == 'ggplot':
                    COLORS = DEFAULT_GGPLOT_COLORS
                else:
                    COLORS = DEFAULT_PLOTLY_COLORS

                num_colors = len(COLORS)
                counter = 0
                for i, cluster in enumerate(cluster_order):
                    if cluster not in cluster_colors:
                        cluster_colors[cluster] = COLORS[counter % num_colors]
                        counter += 1

                # map cluster labels to numeric values
                num_clusters = len(cluster_order)
                #cluster_mapping = dict(
                #    [cluster, i] for i, cluster in enumerate(cluster_order))
                cluster_mapping = dict(
                    [cluster, num_clusters-(i+1)]
                    for i, cluster in enumerate(cluster_order))
                numeric_labels = ann.data['labels'].map(cluster_mapping)

                # construct colorscale
                num_clusters = len(cluster_order)
                colorscale = list(reversed(
                    [((num_clusters-j)/num_clusters,
                      cluster_colors[cluster_order[i]])
                     for i in range(num_clusters) for j in [i, i+1]]))

                trace = ann.get_parent_object()

                # set trace y-axis
                if heatmap_index == 0:
                    trace['yaxis'] = 'y'
                else:
                    trace['yaxis'] = 'y%d' % (heatmap_index + 1)
                trace['xaxis'] = 'x'

                cluster_labels = ann.data.get('clusterlabels', None)
                if cluster_labels is not None:
                    final_labels = \
                            ann.data['labels'].replace(cluster_labels)
                else:
                    final_labels = ann.data['labels']

                z = np.atleast_2d(numeric_labels.values)
                text = np.atleast_2d(final_labels.values)
                trace['x'] = numeric_labels.index
                #trace['y'] = panel.data['matrix'].genes
                trace['z'] = z
                trace['text'] = text
                trace['zmin'] = 0
                trace['zmax'] = num_clusters
                trace['colorscale'] = colorscale

                data.append(trace)

                # calculate relative height (for `domain` property)
                #height_frac = (ann.data['height'] / total_height) * data_height_frac
                height_frac = (ann.data['height'] / heatmap_height_px)

                ### configure y-axis
                title = None
                tickfont = {}

                if heatmap_index == 0:
                    layout_yaxis_key = 'yaxis'
                else:
                    layout_yaxis_key = 'yaxis%d' % (heatmap_index + 1)

                yaxis = go.layout.YAxis(self._default_yaxis)

                yaxis['domain'] = [cur_y - height_frac, cur_y]
                yaxis['title'] = title
                yaxis['autorange'] = 'reversed'

                yaxis['tickvals'] = [0]
                yaxis['ticktext'] = [ann.data['title']]

                yaxis_data = ann.data.get('yaxis', None)
                if yaxis_data is None:
                    yaxis_data = {}
                yaxis.update(yaxis_data)

                layout[layout_yaxis_key] = yaxis
                cur_y -= (height_frac + panelmargin_frac)                

        xaxis = layout['xaxis']
        xaxis['anchor'] = 'y%d' % (len(self.data))

        if xaxis.title.text is None:
            xaxis['title']['text'] = 'Cells (%d)' % data[0].z.shape[1]
        layout['xaxis'] = xaxis

        fig = go.Figure(data=data, layout=layout)

        return fig
