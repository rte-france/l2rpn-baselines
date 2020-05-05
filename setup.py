# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import subprocess
import sys
import setuptools
from setuptools import setup

pkgs = {
    "required": [
        "grid2op[challenge,optional]>=0.8.0"
    ],
    "extras": {
        "docs": [
            "grid2op[docs]"
        ]
    }
}

setup(name='l2rpn_baselines',
      version='0.2.0',
      description='L2RPN Baselines a repository to host ' \
      'baselines for l2rpn competitions.',
      long_description='This repository aims at facilitating ' \
      'the use of state of the art algorithm in coming from the ' \
      'reinforcement learning community or the power system ' \
      'community in the l2rpn competitions. It  also provides ' \
      'some usefull function to make life or participants to the ' \
      'l2rpn competitions easier.',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
          "Intended Audience :: Developers",
          "Intended Audience :: Education",
          "Intended Audience :: Science/Research",
          "Natural Language :: English"
      ],
      keywords='ML powergrid optmization RL power-systems',
      author='Benjamin DONNOT',
      author_email='benjamin.donnot@rte-france.com',
      url="https://github.com/BDonnot/L2RPN_Baselines",
      license='MPL',
      packages=setuptools.find_packages(),
      include_package_data=True,
      install_requires=pkgs["required"],
      extras_require=pkgs["extras"],
      zip_safe=False,
      entry_points={}
)
