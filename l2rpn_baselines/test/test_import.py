# test that the baselines can be imported

import unittest
import l2rpn_baselines


class TestImport():
    def test_import(self):
        module_name = self.load_module()
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


if __name__ == "__main__":
    unittest.main()