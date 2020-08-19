# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import warnings
import tensorflow as tf
import tensorflow.summary as tfs
import numpy as np

def tf_limit_gpu_usage():
    WARN_GPU_CONF = "Cannot configure tensorflow GPU memory growth property"
    try:
        devices = tf.config.list_physical_devices('GPU')
        if len(devices) > 0:
            tf.config.experimental.set_memory_growth(devices[0], True)
    except AttributeError:
        try:
            devices = tf.config.experimental.list_physical_devices('GPU')
            if len(devices) > 0:
                tf.config.experimental.set_memory_growth(devices[0], True)
        except Exception:
            warnings.warn(WARN_GPU_CONF)
    except Exception:
        warnings.warn(WARN_GPU_CONF)

class TensorboardLogger(object):
    def __init__(self, name, logdir):
        self._writer = tfs.create_file_writer(logdir, name=name)
        self._scalars = {}
        self._mean_scalars = {}

    def clear(self):
        self._scalars = {}
        self._mean_scalars = {}

    def scalar(self, name, value):
        self._scalars[name] = value

    def mean_scalar(self, name, value, length=100):
        scalar_li = self._mean_scalars.get(name, [])
        scalar_li.append(value)
        if len(scalar_li) > length:
            scalar_li.pop(0)
        self._mean_scalars[name] = scalar_li

    def write(self, step):
        with self._writer.as_default():
            # Handle pure scalars
            for key, val in self._scalars.items():
                tfs.scalar(key, val, step)
            # Handle running averages
            for key, val_li in self._mean_scalars.items():
                val = np.mean(val_li)
                tfs.scalar(key, val, step)
        
        
