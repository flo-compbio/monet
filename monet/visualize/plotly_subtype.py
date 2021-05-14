# Copyright (c) 2020, 2021 Florian Wagner

# This file is part of Monet.

from typing import Dict
from collections import UserDict
import copy

import collections.abc


def recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class PlotlySubtype(UserDict):
    """Abstract base class for subclassing Plotly classes."""

    _subtype_props = None  # replace with set containing subtype properties
    _parent_type = None  # replace with parent Plotly class
    _default_values = None  # replace with dictionary containing default values

    def __init__(self, data: Dict = None, **kwargs):

        if data is None:
            data = {}

        # initialize dictionary with default values
        final_data = copy.deepcopy(self._default_values)
        #final_data.update(data)
        recursive_update(final_data, data)

        # let specified keywords override values in `data`
        #final_data.update(kwargs)
        recursive_update(final_data, kwargs)

        # make a copy
        dummy_data = copy.deepcopy(final_data)

        # remove all subtype properties from copy
        for prop in self._subtype_props:
            dummy_data.pop(prop, None)

        # test if the remaining properties are kosher, according to plotly
        # this raises a ValueError if any of the properties are invalid
        # (Plotly's error message lists all valid properties)
        self._dummy = self._parent_type(dummy_data)

        # test if the subtype properties are kosher
        invalid_keys = []
        for k in final_data.keys():
            if (k not in dummy_data) and (k not in self._subtype_props):
                invalid_keys.append(k)

        if invalid_keys:
            error_msg = 'Invalid %s subtype properties specified: %s' \
                    % (self.__class__.__name__, str(tuple(invalid_keys)))
            raise ValueError(error_msg)

        super().__init__(final_data)


    def __setitem__(self, key, value):
        # test if the key is valid
        try:
            # first try if key is valid for parent type
            self._dummy[key] = value
        except ValueError as error:
            # if not, try if key is a valid subtype property
            if key not in self._subtype_props:
                error_msg = str(error)
                subtype_str = '\n        '.join(self._subtype_props)
                error_msg = error_msg + '\n    Valid subtype properties:\n        ' + subtype_str
                raise ValueError(error_msg)

        # the key is valid, add it to the data
        self.data[key] = value


    def get_parent_object(self):
        parent_dict = {}
        for key, val in self.data.items():
            if key not in self._subtype_props:
                parent_dict[key] = val
        parent_obj = self._parent_type(parent_dict)
        return parent_obj
