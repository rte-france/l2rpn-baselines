CurriculumAgent
===============

The CurriculumAgent baseline is a Reinforcement Learning Agent designed to learn from and act within the  
[Grid2Op Environments](https://grid2op.readthedocs.io/en/latest/). The overall functionallity of the different modules
can be found in the package [curriculumagent](https://github.com/FraunhoferIEE/curriculumagent). The CurriculumAgent is 
a cleanup and improved version of the[NeurIPS 2020 Competition Agent by binbinchen](https://github.com/AsprinChina/L2RPN_NIPS_2020_a_PPO_Solution),
The agent is build to extract action sets of the Grid2Op Environment and then use rule-based agent to train
a Reinforcement Learning agent. We explain each step in more detail in our paper. 

When using the CurriculumAgent, please cite our paper with.
```
@article{lehna_managing_2023,
	title = {Managing power grids through topology actions: A comparative study between advanced rule-based and reinforcement learning agents},
	issn = {2666-5468},
	url = {https://www.sciencedirect.com/science/article/pii/S2666546823000484},
	doi = {https://doi.org/10.1016/j.egyai.2023.100276},
	pages = {100276},
	journaltitle = {Energy and {AI}},
	author = {Lehna, Malte and Viebahn, Jan and Marot, Antoine and Tomforde, Sven and Scholz, Christoph},
	date = {2023},
}
```
Usage/Documentation
-------------------

The baseline was primarily created to run on the IEEE14 bus network. However, with some small changes, the baseline can
be reconfigured for the other grids. Checkout the  [github](https://github.com/FraunhoferIEE/curriculumagent)
page for updates as well as the citation. Please checkout the original project as well.


License
-------

```
Copyright (c) 2022 EI Innovation Lab, Huawei Cloud, Huawei Technologies and Fraunhofer IEE
The code is subject to the terms of Mozilla Public License (MPL) v2.0.
Commercial use is NOT allowed.
```

Please take a look at the [LICENSE](https://github.com/FraunhoferIEE/curriculumagent/blob/master/LICENSE) file for a
full copy of the MPL license.
