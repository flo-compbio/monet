# Author: Florian Wagner <florian.compbio@gmail.com>
# Copyright (c) 2020 Florian Wagner
#
# This file is part of Monet.

from typing import List, Tuple

import numpy as np


DEFAULT_PLOTLY_COLORS = [
    'rgb(31, 119, 180)', 'rgb(255, 127, 14)',
    'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
    'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
    'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
    'rgb(188, 189, 34)', 'rgb(23, 190, 207)',
    'rgb(99, 110, 250)', 'rgb(239, 85, 59)',
    'rgb(0, 204, 150)', 'rgb(171, 99, 250)',
    'rgb(255, 161, 90)', 'rgb(25, 211, 243)',
    'rgb(255, 102, 146)', 'rgb(182, 232, 128)',
    'rgb(255, 151, 255)', 'rgb(254, 203, 82)']

ACCESSIBLE_COLORS = [
    'rgb(0,114,178)',  # Blue
    'rgb(230,159,0)',  # Orange
    'rgb(204,121,167)',  # Reddish purple
    'rgb(86,180,223)',  # Sky blue    
    'rgb(213,94,0)',  # Vermillion    
]

DEFAULT_GGPLOT_COLORS = [
    # from hue_pal(2)
    '#F8766D', '#00BFC4',

    # from hue_pal(4)
    '#7CAE00', '#C77CFF',

    # from hue_pal(8)
    '#CD9600', '#00A9FF', '#00BE67', '#FF61CC',

    # from hue_pal(16)
    '#E68613', '#00B8E7', '#0CB702', '#ED68ED',
    '#ABA300', '#8494FF', '#00C19A', '#FF68A1',

    # from hue_pal(32)
    '#F07E4C', '#00BCD7', '#59B300', '#DD71FA',
    '#BD9D00', '#49A0FF', '#00C082', '#FF63B8',
    '#DA8E00', '#00B1F4', '#00BB46', '#F862DE',
    '#96A900', '#AA88FF', '#00C0B0', '#FE6E88',
]

def load_colorscale(fpath: str) -> List[Tuple[float, str]]:
    data = np.loadtxt(fpath, delimiter='\t', dtype=np.float64)
    rgb = np.int64(data[:, 1:])
    n = data.shape[0]
    colorscale = []
    for i in range(n):
        colorscale.append(
            (i / float(n-1),
            'rgb(%d, %d, %d)' % (rgb[i, 0], rgb[i, 1], rgb[i, 2])))
    return colorscale
