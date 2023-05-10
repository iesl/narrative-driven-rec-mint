"""
Miscellaneous utilities to read and work with the json files and such.
Stuff multiple functions use.
"""
import sys
import os
import errno
import re
import codecs, json, random
import copy

import pandas as pd


def create_dir(dir_name):
    """
    Create the directory whose name is passed.
    :param dir_name: String saying the name of directory to create.
    :return: None.
    """
    # Create output directory if it doesnt exist.
    try:
        os.makedirs(dir_name)
        print('Created: {}.'.format(dir_name))
    except OSError as ose:
        # For the case of *file* by name of out_dir existing
        if (not os.path.isdir(dir_name)) and (ose.errno == errno.EEXIST):
            sys.stderr.write('IO ERROR: Could not create output directory\n')
            sys.exit(1)
        # If its something else you don't know; report it and exit.
        if ose.errno != errno.EEXIST:
            sys.stderr.write('OS ERROR: {:d}: {:s}: {:s}\n'.format(ose.errno,
                                                                   ose.strerror,
                                                                   dir_name))
            sys.exit(1)
