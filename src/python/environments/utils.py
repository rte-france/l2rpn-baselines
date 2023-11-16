from collections import OrderedDict

node_data_fields = OrderedDict(
    {
        "substation": [
            "id"
        ],  # All fields: {'id': 1, 'type': 'substation', 'name': 'sub_1', 'cooldown': 0}
        "bus": [
            "v",
            "theta",
            "id",
        ],  # All fields: {'id': 2, 'global_id': 2, 'local_id': 1, 'type': 'bus', 'connected': True, 'v': 142.1, 'theta': -4.190119}
        "load": [
            "id"
        ],  # {'id': 8, 'type': 'load', 'name': 'load_11_8', 'connected': True}
        "gen": [
            # "id",
            # "target_dispatch",
            # "actual_dispath",
            # "gen_p_before_curtal",
            # "curtalment_mw",
            # "curtailement",
            # "curtailment_limit",
            # "gen_margin_up",
            # "gen_margin_down",
            "difference_dispatch",
            "target_dispatch",
            "actual_dispatch",
        ],  # {'id': 3, 'type': 'gen', 'name': 'gen_5_3', 'connected': True, 'target_dispatch': 0.0, 'actual_dispatch': 0.0, 'gen_p_before_curtail': 0.0, 'curtailment_mw': 0.0, 'curtailment': 0.0, 'curtailment_limit': 1.0, 'gen_margin_up': 0.0, 'gen_margin_down': 0.0}
        "line": [
            "id",
            "rho",
        ],  # {'id': 1, 'type': 'line', 'name': '0_4_1', 'rho': 0.36042336, 'connected': True, 'timestep_overflow': 0, 'time_before_cooldown_line': 0, 'time_next_maintenance': -1, 'duration_next_maintenance': 0}
        "shunt": [
            "id"
        ],  # {'id': 0, 'type': 'shunt', 'name': 'shunt_8_0', 'connected': True}
    }
)

edge_data_fields = OrderedDict(
    {
        "bus_to_substation": {},
        "load_to_bus": {
            "id",
            "p",
            "q",
            "v",
            "theta",
        },  # {'id': 0, 'type': 'load_to_bus', 'p': 21.9, 'q': 15.4, 'v': 142.1, 'theta': -1.4930121}
        "gen_to_bus": {
            "id",
            "p",
            "q",
            "v",
            "theta",
        },  # {'id': 0, 'type': 'gen_to_bus', 'p': -81.4, 'q': -19.496038, 'v': 142.1, 'theta': -1.4930121}
        "line_to_bus": {
            "id",
            "p",
            "q",
            "v",
            "theta",
            "a",
        },  # {'id': 0, 'type': 'line_to_bus', 'p': 42.346096, 'q': -16.060501, 'v': 142.1, 'a': 184.01027, 'side': 'or', 'theta': 0.0}
        "shunt_to_bus": {
            "id",
            "p",
            "v",
            "q",
        },  # {'id': 0, 'type': 'shunt_to_bus', 'p': -6.938894e-16, 'q': -21.208096, 'v': 21.13022}
    }
)
