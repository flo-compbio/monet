# Author: Florian Wagner <florian.wagner@uchicago.edu>
# Copyright (c) 2020 Florian Wagner
#
# This file is part of Monet.

from pandas.testing import assert_frame_equal

def assert_frame_not_equal(*args, **kwargs):
    try:
        assert_frame_equal(*args, **kwargs)
    except AssertionError:
        pass
    else:
        raise AssertionError
