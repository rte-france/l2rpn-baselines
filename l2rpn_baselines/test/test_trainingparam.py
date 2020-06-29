# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

# test that the baselines can be imported
import os
import unittest
import tempfile
from l2rpn_baselines.utils import TrainingParam
import pdb


class TestTrainingParam(unittest.TestCase):
    def test_save(self):
        tp = TrainingParam()
        tmp_dir = tempfile.mkdtemp()
        tp.save_as_json(tmp_dir, "test.json")

    def test_loadback(self):
        tp = TrainingParam()
        tmp_dir = tempfile.mkdtemp()
        tp.save_as_json(tmp_dir, "test.json")

        tp2 = TrainingParam.from_json(os.path.join(tmp_dir, "test.json"))
        assert tp2 == tp

    def test_loadback_modified(self):
        for el in TrainingParam._int_attr:
            self._aux_test_attr(el, 1)
            self._aux_test_attr(el, None)
        for el in TrainingParam._float_attr:
            self._aux_test_attr(el, 1.)
            self._aux_test_attr(el, None)

    def _aux_test_attr(self, attr, val):
        """
        test that i can modify an attribut and then load the training parameters the correct way
        """
        tp = TrainingParam()
        setattr(tp, attr, val)
        tmp_dir = tempfile.mkdtemp()
        tp.save_as_json(tmp_dir, "test.json")
        tp2 = TrainingParam.from_json(os.path.join(tmp_dir, "test.json"))
        assert tp2 == tp, "error for attributes {}".format(attr)

    def test_get_epsilon(self):
        tp = TrainingParam()
        tp.final_epsilon = None
        eps = tp.get_next_epsilon(1)
        assert eps == 0.
        tp.final_epsilon = 0.01
        tp.initial_epsilon = None
        eps = tp.get_next_epsilon(1)
        assert eps == 0.
        tp.initial_epsilon = 0.01
        tp.final_epsilon = 0.01
        eps = tp.get_next_epsilon(1)
        assert eps == 0.01


if __name__ == "__main__":
    unittest.main()