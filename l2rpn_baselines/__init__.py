# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import copy
import traceback

all_baselines_li = [
    "Template",
    "DoubleDuelingDQN",
    "DoubleDuelingRDQN",
    "DoNothing",
    "SliceRDQN",
    "DeepQSimple",
    "DuelQSimple",
    "LeapNetEncoded",
    # Backward compatibility
    "SACOld",
    # contribution
    "PandapowerOPFAgent",
    "Geirina",
    "AsynchronousActorCritic",
    "Kaist",
    # utilitary scripts
    "utils",
    "all_baselines",
    "available_baselines"
]
__all__ = ["Template", "DoNothing"]
__import_error_dict = {}

__version__ = "0.5.0"

try:
    import l2rpn_baselines.DoubleDuelingDQN
    __all__.append("DoubleDuelingDQN")
except ImportError as exc_:
    __import_error_dict["DoubleDuelingDQN"] = exc_, traceback.format_exc()

try:
    import l2rpn_baselines.SliceRDQN
    __all__.append("SliceRDQN")
except ImportError as exc_:
    __import_error_dict["SliceRDQN"] = exc_, traceback.format_exc()

try:
    import l2rpn_baselines.DoubleDuelingRDQN
    __all__.append("DoubleDuelingRDQN")
except ImportError as exc_:
    __import_error_dict["DoubleDuelingRDQN"] = exc_, traceback.format_exc()

try:
    import l2rpn_baselines.DeepQSimple
    __all__.append("DeepQSimple")
except ImportError as exc_:
    __import_error_dict["DeepQSimple"] = exc_, traceback.format_exc()

try:
    import l2rpn_baselines.DuelQSimple
    __all__.append("DuelQSimple")
except ImportError as exc_:
    __import_error_dict["DuelQSimple"] = exc_, traceback.format_exc()

try:
    import l2rpn_baselines.LeapNetEncoded
    __all__.append("LeapNetEncoded")
except ImportError as exc_:
    __import_error_dict["LeapNetEncoded"] = exc_, traceback.format_exc()


# Backward compatibility
try:
    import l2rpn_baselines.SACOld
    __all__.append("SACOld")
except ImportError as exc_:
    __import_error_dict["SACOld"] = exc_, traceback.format_exc()

# contribution
try:
    import l2rpn_baselines.PandapowerOPFAgent
    __all__.append("PandapowerOPFAgent")
except ImportError as exc_:
    __import_error_dict["PandapowerOPFAgent"] = exc_, traceback.format_exc()

try:
    # TODO if i import them, then everything else crashes, ask them to put "tf.disable_v2_behavior()"
    #  in their code somewhere
    # import l2rpn_baselines.Geirina
    __all__.append("Geirina")
except ImportError as exc_:
    __import_error_dict["Geirina"] = exc_, traceback.format_exc()

try:
    # TODO if i import them, then everything else crashes, ask them to put "tf.disable_v2_behavior()"
    #  in their code somewhere
    # import l2rpn_baselines.AsynchronousActorCritic
    __all__.append("AsynchronousActorCritic")
except ImportError as exc_:
    __import_error_dict["AsynchronousActorCritic"] = exc_, traceback.format_exc()

try:
    # TODO may cause issue with multi processing
    # import l2rpn_baselines.Kaist
    # __all__.append("Kaist")
    pass
except ImportError as exc_:
    __import_error_dict["Kaist"] = exc_, traceback.format_exc()


def all_baselines():
    """
    return the available baselines

    Examples
    --------
    You can use it as:

    .. code-block:: python

        import l2rpn_baselines
        avail_ = l2rpn_baselines.all_baselines()
        print("The list of all available baselines is:")
        print("\n".join(avail_))

    """
    return copy.deepcopy(all_baselines_li)


def available_baselines():
    """
    return the baselines available on your system. You might need to install extra dependencies for
    baselines not listed here.

    You might want to check the :func:`get_import_error` for more information about what package
    you need to install.

    Examples
    --------
    You can use it as:

    .. code-block:: python

        import l2rpn_baselines
        avail_ = l2rpn_baselines.available_baselines()
        print("Baseline that i can use on my systems are:")
        print("\n".join(avail_))

    """
    return copy.deepcopy(__all__)


def get_import_error(baseline_name):
    """
    get the error encounter when trying to import a baselines

    Parameters
    ----------
    baseline_name:
        Name of the baseline you want to get more about possible loading errors

    Returns
    -------
    A tuple with:

    - first element the error (object of type ImportError)
    - second element: the import traceback (str)

    Examples
    --------
    You can use it as:

    .. code-block:: python

        import l2rpn_baselines
        import_error, error_traceback = l2rpn_baselines.get_import_error("Kaist")
        print(error_traceback)

        # can plot something like (only if torch is not installed on your machine)
        # Traceback (most recent call last):
        #   File "l2rpn_baselines/__init__.py", line 103, in <module>
        #     import l2rpn_baselines.Kaist
        #   File "l2rpn_baselines/Kaist/__init__.py", line 6, in <module>
        #     from l2rpn_baselines.Kaist.Kaist import Kaist
        #   File "l2rpn_baselines/Kaist/Kaist.py", line 3, in <module>
        #     import torch
        # ModuleNotFoundError: No module named 'torch'

    """
    if baseline_name in __all__:
        return None, ""

    if not baseline_name in __import_error_dict:
        raise RuntimeError("\"{}\" does not name a baseline. List of supported baseline for "
                           "this function are \"{}\"."
                           "".format(baseline_name, sorted(list(__import_error_dict.keys()))))

    return __import_error_dict[baseline_name]