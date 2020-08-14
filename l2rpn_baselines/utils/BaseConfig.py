# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import os
import json
import numpy as np
from collections.abc import Iterable


class BaseConfig(object):
    """
    This class provides an easy way to save and restore, as json, 
    the configurations of your neural networks or training parameters.

    It is recommended to overload this class for each specific use case.
    """

    _int_attr = []
    _float_attr = []
    _str_attr = []
    _bool_attr = []
    _list_float = []
    _list_str = []
    _list_int = []
    _list_bool = []

    __attr_serialize = {
        "_int_attr": int,
        "_float_attr": float,
        "_str_attr": str,
        "_bool_attr": bool,
        "_list_int": int,
        "_list_float": float,
        "_list_str": str,
        "_list_bool": bool
    }

    @staticmethod
    def __list_to_json(obj, type_):
        if isinstance(obj, type_):
            res = obj
        elif isinstance(obj, np.ndarray):
            if len(obj.shape) == 1:
                res = [type_(el) for el in obj]
            else:
                res= []
                for el in obj:
                    res.append(BaseConfig.__list_to_json(el, type_))
        elif isinstance(obj, Iterable):
            res = [BaseConfig.__list_to_json(el, type_) for el in obj]
        else:
            res = type_(obj)
        return res

    @staticmethod
    def __serialize(json, type_):
        if isinstance(json, type_):
            res = json
        elif isinstance(json, Iterable):
            res = []
            for el in json:
                res.append(BaseConfig.__list_to_json(obj=el,type_=type_))
        else:
            res = type_(json)
        return res

    @classmethod
    def from_dict(cls, tmp):
        """Load an instance from a dictionnary"""
        res = cls()

        for attr_cat, attr_type in BaseConfig.__attr_serialize.items():
            for attr_nm in getattr(cls, attr_cat):
                if attr_nm in tmp:
                    attr_val = BaseConfig.__serialize(tmp[attr_nm], attr_type)
                    setattr(res, attr_nm, attr_val)
        return res

    @classmethod
    def from_json_file(cls, json_path):
        """Load an instance from a json file"""

        if not os.path.exists(json_path):
            err_msg = "File does not exists: \"{}\"".format(json_path)
            raise FileNotFoundError(err_msg)

        with open(json_path, "r") as f:
            dict_ = json.load(f)

        return cls.from_dict(dict_)

    def to_dict(self):
        """Convert this instance to a dictionnary"""
        res = {}
        for attr_cat, attr_type in BaseConfig.__attr_serialize.items():
            for attr_nm in getattr(self, attr_cat):
                attr_val = getattr(self, attr_nm)
                res[attr_nm] = self.__serialize(attr_val, attr_type)
        return res

    def to_json_file(self, json_path):
        """Save this instance as a json file"""

        res = self.to_dict()
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(res, fp=f, indent=4, sort_keys=True)
