import numpy as np    

def lines_q_len(action_space):
    all_actions = [action_space({})]
    if "_set_line_status" in action_space._template_act.attr_list_vect:
        all_actions += action_space.get_all_unitary_line_set(action_space)

    if "_switch_line_status" in action_space._template_act.attr_list_vect:
        all_actions += action_space.get_all_unitary_line_change(action_space)

    return len(all_actions)

def topo_q_len(action_space):
    all_actions = [action_space({})]
    if "_set_topo_vect" in action_space._template_act.attr_list_vect:
        all_actions += action_space.get_all_unitary_topologies_set(action_space)

    if "_change_bus_vect" in action_space._template_act.attr_list_vect:
        all_actions += action_space.get_all_unitary_topologies_change(action_space)

    return len(all_actions)

def disp_q_len(action_space):
    all_actions = [action_space({})]
    if "_redispatch" in action_space._template_act.attr_list_vect:
        all_actions += action_space.get_all_unitary_redispatch(action_space)

    return len(all_actions)

def shape_obs(observation_space):
    dims = np.array([
        observation_space.n_line,
        observation_space.n_gen,
        observation_space.n_load,
        observation_space.n_sub,
        5
    ])
    return (25, np.max(dims))

def to_pad_vect(inputv, pad_w, pad_v = 0.0, scale_v = 1.0):
    v = np.asarray(inputv)
    v = v / scale_v
    vsafe = np.nan_to_num(v, nan=pad_v, posinf=pad_v, neginf=pad_v)
    padder = (0, pad_w - len(vsafe))
    vpad = np.pad(vsafe, padder, constant_values=pad_v)
    return vpad.astype(np.float32)

def convert_obs_pad(obs, bias=0.0):
    # Store some shortcuts
    topo = obs.topo_vect
    g_pos = obs.gen_pos_topo_vect
    l_pos = obs.load_pos_topo_vect
    lor_pos = obs.line_or_pos_topo_vect
    lex_pos = obs.line_ex_pos_topo_vect
    
    # Find longest dim
    dims = np.array([obs.n_line, obs.n_gen, obs.n_load, obs.n_sub, 5])
    pad_w = np.max(dims)

    # Get time data
    time_li = [obs.month / 12.0, obs.day / 31.0, obs.day_of_week / 7.0,
               obs.hour_of_day / 24.0, obs.minute_of_hour / 60.0]
    time_v = to_pad_vect(time_li, pad_w)
    time_line_cd = to_pad_vect(obs.time_before_cooldown_line, pad_w, pad_v=-1.0, scale_v=10.0)
    time_line_nm = to_pad_vect(obs.time_next_maintenance, pad_w, scale_v=10.0)
    time_sub_cd = to_pad_vect(obs.time_before_cooldown_sub, pad_w, pad_v=-1.0, scale_v=10.0)
    
    # Get generators info
    g_p = to_pad_vect(obs.prod_p, pad_w, scale_v=1000.0)
    g_q = to_pad_vect(obs.prod_q, pad_w, scale_v=1000.0)
    g_v = to_pad_vect(obs.prod_v, pad_w, scale_v=1000.0)
    g_tr = to_pad_vect(obs.target_dispatch, pad_w, scale_v=150.0)
    g_ar = to_pad_vect(obs.actual_dispatch, pad_w, scale_v=150.0)
    g_cost = to_pad_vect(obs.gen_cost_per_MW, pad_w, pad_v=0.0, scale_v=1.0)
    g_buses = np.zeros(obs.n_gen)
    for gen_id in range(obs.n_gen):
        g_buses[gen_id] = topo[g_pos[gen_id]]
        if g_buses[gen_id] <= 0.0:
            g_buses[gen_id] = 0.0
    g_bus = to_pad_vect(g_buses, pad_w, pad_v=-1.0, scale_v=3.0)

    # Get loads info
    l_p = to_pad_vect(obs.load_p, pad_w, scale_v=1000.0)
    l_q = to_pad_vect(obs.load_q, pad_w, scale_v=1000.0)
    l_v = to_pad_vect(obs.load_v, pad_w, scale_v=1000.0)
    l_buses = np.zeros(obs.n_load)
    for load_id in range(obs.n_load):
        l_buses[load_id] = topo[l_pos[load_id]]
        if l_buses[load_id] <= 0.0:
            l_buses[load_id] = 0.0
    l_bus = to_pad_vect(l_buses, pad_w, pad_v=-1.0, scale_v=3.0)

    # Get lines origin info
    or_p = to_pad_vect(obs.p_or, pad_w, scale_v=1000.0)
    or_q = to_pad_vect(obs.q_or, pad_w, scale_v=1000.0)
    or_v = to_pad_vect(obs.v_or, pad_w, scale_v=1000.0)
    or_buses = np.zeros(obs.n_line)
    for line_id in range(obs.n_line):
        or_buses[line_id] = topo[lor_pos[line_id]]
        if or_buses[line_id] <= 0.0:
            or_buses[line_id] = 0.0
    or_bus = to_pad_vect(or_buses, pad_w, pad_v=-1.0, scale_v=3.0)
    or_rho = to_pad_vect(obs.rho, pad_w, pad_v=-1.0)
    
    # Get extremities origin info
    ex_p = to_pad_vect(obs.p_ex, pad_w, scale_v=1000.0)
    ex_q = to_pad_vect(obs.q_ex, pad_w, scale_v=1000.0)
    ex_v = to_pad_vect(obs.v_ex, pad_w, scale_v=1000.0)
    ex_buses = np.zeros(obs.n_line)
    for line_id in range(obs.n_line):
        ex_buses[line_id] = topo[lex_pos[line_id]]
        if ex_buses[line_id] <= 0.0:
            ex_buses[line_id] = 0.0
    ex_bus = to_pad_vect(ex_buses, pad_w, pad_v=-1.0, scale_v=3.0)
    ex_rho = to_pad_vect(obs.rho, pad_w, pad_v=-1.0)

    res = np.stack([
        # [0;3] Time
        time_v, time_line_cd, time_sub_cd, time_line_nm,
        # [4;10] Gens
        g_p, g_q, g_v, g_ar, g_tr, g_bus, g_cost,
        # [11;14] Loads
        l_p, l_q, l_v, l_bus,
        # [15;19] Origins
        or_p, or_q, or_v, or_bus, or_rho,
        # [20;24] Extremities
        ex_p, ex_q, ex_v, ex_bus, ex_rho
    ])
    return res + bias
