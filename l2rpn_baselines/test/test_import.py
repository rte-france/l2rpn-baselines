# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

# test that the baselines can be imported

import unittest
import l2rpn_baselines


class TestImport(object):
    def test_import(self):
        module_name = self.load_module()
        exec(f"import l2rpn_baselines.{module_name}")
        exec(f"from l2rpn_baselines.{module_name} import {module_name}")
        exec(f"from l2rpn_baselines.{module_name} import evaluate")
        assert 1+1 == 2


class TestDN(TestImport, unittest.TestCase):
    def load_module(self):
        return "DoNothing"


class TestTemplate(TestImport, unittest.TestCase):
    def load_module(self):
        return "Template"


class TestD3QN(TestImport, unittest.TestCase):
    def load_module(self):
        return "DoubleDuelingDQN"


class TestD3RQN(TestImport, unittest.TestCase):
    def load_module(self):
        return "DoubleDuelingRDQN"


class TestDeepQSimple(TestImport, unittest.TestCase):
    def load_module(self):
        return "DeepQSimple"


class TestSACOld(TestImport, unittest.TestCase):
    def load_module(self):
        return "SACOld"


class TestLeapNetEnc(TestImport, unittest.TestCase):
    def load_module(self):
        return "LeapNetEncoded"


class TestDuelQSimple(TestImport, unittest.TestCase):
    def load_module(self):
        return "DuelQSimple"


class TestSliceRDQN(TestImport, unittest.TestCase):
    def load_module(self):
        return "SliceRDQN"


class TestDuelQLeapNet(TestImport, unittest.TestCase):
    def load_module(self):
        return "DuelQLeapNet"


class TestPandapowerOPFAgent(TestImport, unittest.TestCase):
    def load_module(self):
        return "PandapowerOPFAgent"


class TestKaist(TestImport, unittest.TestCase):
    def load_module(self):
        return "Kaist"


class TestExpertAgent(TestImport, unittest.TestCase):
    def load_module(self):
        return "ExpertAgent"

class TestPPOSB3(TestImport, unittest.TestCase):
    def load_module(self):
        return "PPO_SB3"

class TestPPOSB3(TestImport, unittest.TestCase):
    def load_module(self):
        return "PPO_RLLIB"


# because it deactivates the eager mode
# class TestPandapowerGeirina(TestImport, unittest.TestCase):
#     def load_module(self):
#         return "Geirina"


# class TestAsynchronousActorCritic(TestImport, unittest.TestCase):
#     def load_module(self):
#         return "AsynchronousActorCritic"


if __name__ == "__main__":
    unittest.main()