import warnings
import numpy as np
import math
import networkx as nx

def sac_size_obs(observation_space):
    dims = np.array([
        # Time
        #5, # Timestamp
        4 * observation_space.n_line, # lines cd
        observation_space.n_sub, # Sub cd
        # Gen
        observation_space.n_gen * 4,
        # Load
        observation_space.n_load * 4,
        # Line origins
        observation_space.n_line * 6,
        # Line extremities
        observation_space.n_line * 6
    ])
    return np.sum(dims)

def sac_bridge_obs(obs):
    # Store some obs shortcuts
    n_sub = obs.n_sub
    n_line = obs.n_line
    topo = obs.topo_vect
    or_topo = obs.line_or_pos_topo_vect
    ex_topo = obs.line_ex_pos_topo_vect
    or_sub = obs.line_or_to_subid
    ex_sub = obs.line_ex_to_subid

    # Create a graph of vertices
    # Use one vertex per substation per bus
    G = nx.Graph()

    # Set lines edges for current bus
    for line_idx in range(n_line):
        # Skip if line is disconnected
        if obs.line_status[line_idx] is False:
            continue
        # Get substation index for current line
        lor_sub = or_sub[line_idx]
        lex_sub = ex_sub[line_idx]
        # Get the buses for current line
        lor_bus = topo[or_topo[line_idx]]
        lex_bus = topo[ex_topo[line_idx]]

        # Skip if one end of the line is disconnected
        if lor_bus <= 0 or lex_bus <= 0:
            continue

        # Compute edge vertices indices for current graph
        left_v =  lor_sub + (lor_bus - 1) * n_sub
        right_v = lex_sub + (lex_bus - 1) * n_sub

        # Register edge in graph
        G.add_edge(left_v, right_v, line_id=line_idx)

    # Declare mask
    lines_bridges = np.zeros(n_line)
    # Find the bridges
    bridges = list(nx.bridges(G))
    for bl, br in bridges:
        line_bridge = G[bl][br]["line_id"]

        lor_bus = topo[or_topo[line_bridge]]
        lex_bus = topo[ex_topo[line_bridge]]

        lor_sub = or_sub[line_bridge]
        lor_start = np.sum(obs.sub_info[:lor_sub])
        lor_end = lor_start + obs.sub_info[lor_sub]
        lor_sub_topo = obs.topo_vect[lor_start:lor_end]
        lor_sub_topo = lor_sub_topo[lor_sub_topo != -1]
        if lor_bus != lex_bus and np.sum(lor_sub_topo == 2) == 1:
            continue

        lex_sub = ex_sub[line_bridge]
        lex_start = np.sum(obs.sub_info[:lex_sub])
        lex_end = lex_start + obs.sub_info[lex_sub]
        lex_sub_topo = obs.topo_vect[lex_start:lex_end]
        lex_sub_topo = lex_sub_topo[lex_sub_topo != -1]
        if lor_bus != lex_bus and np.sum(lex_sub_topo == 2) == 1:
            continue

        lines_bridges[line_bridge] = 1.0

    return lines_bridges

def sac_split_obs(obs):
    splitted_mask = np.zeros(obs.n_sub)
    for sub_id in range(obs.n_sub):
        sub_start = np.sum(obs.sub_info[:sub_id])
        sub_end = sub_start + obs.sub_info[sub_id]
        sub_topo = obs.topo_vect[sub_start:sub_end]
        sub_topo = sub_topo[sub_topo != -1]
        if np.any(sub_topo[1:] != sub_topo[0]):
            splitted_mask[sub_id] = 1.0

    return splitted_mask

def to_safe_vect(input_v, pad_v = 0.0):
    v = np.asarray(input_v).astype(np.float32)
    vsafe = np.nan_to_num(v, nan=pad_v, posinf=pad_v, neginf=pad_v)
    return vsafe

def to_norm_vect(input_v, pad_v = 0.0):
    vsafe = to_safe_vect(input_v)
    vmean = np.mean(vsafe)
    vstd = np.std(vsafe)
    vnorm = vsafe - vmean
    if vstd != 0.0:
        vnorm /= vstd
    return vnorm

def sac_convert_obs(obs, bias=0.0):
    # Store some shortcuts
    topo = obs.topo_vect
    g_pos = obs.gen_pos_topo_vect
    l_pos = obs.load_pos_topo_vect
    lor_pos = obs.line_or_pos_topo_vect
    lex_pos = obs.line_ex_pos_topo_vect

    vect_fn = to_safe_vect
    #vect_fn = to_norm_vect

    # Get time data
    time_li = [obs.month / 12.0, obs.day / 31.0, obs.day_of_week / 7.0,
               obs.hour_of_day / 24.0, obs.minute_of_hour / 60.0]
    time_v = np.array(time_li)
    time_line_cd = vect_fn(obs.time_before_cooldown_line)
    time_line_nm = vect_fn(obs.time_next_maintenance)
    time_line_dm = vect_fn(obs.duration_next_maintenance)
    time_line_overflow = vect_fn(obs.timestep_overflow)
    time_sub_cd = vect_fn(obs.time_before_cooldown_sub)

    # Get generators info
    g_p = vect_fn(obs.prod_p)
    g_q = vect_fn(obs.prod_q)
    g_v = vect_fn(obs.prod_v)
    g_tr = vect_fn(obs.target_dispatch)
    g_ar = vect_fn(obs.actual_dispatch)
    g_cost = vect_fn(obs.gen_cost_per_MW)
    g_buses = np.zeros(obs.n_gen)
    for gen_id in range(obs.n_gen):
        g_buses[gen_id] = topo[g_pos[gen_id]] * 1.0
    g_bus = vect_fn(g_buses, pad_v=0.0)

    # Get loads info
    l_p = vect_fn(obs.load_p)
    l_q = vect_fn(obs.load_q)
    l_v = vect_fn(obs.load_v)
    l_buses = np.zeros(obs.n_load)
    for load_id in range(obs.n_load):
        l_buses[load_id] = topo[l_pos[load_id]] * 1.0
    l_bus = vect_fn(l_buses, pad_v=0.0)

    # Get lines origin info
    or_p = vect_fn(obs.p_or)
    or_q = vect_fn(obs.q_or)
    or_a = vect_fn(obs.a_or)
    or_v = vect_fn(obs.v_or)
    or_buses = np.zeros(obs.n_line)
    for line_id in range(obs.n_line):
        or_buses[line_id] = topo[lor_pos[line_id]] * 1.0
    or_bus = vect_fn(or_buses)
    or_rho = vect_fn(obs.rho)
    
    # Get extremities origin info
    ex_p = vect_fn(obs.p_ex)
    ex_q = vect_fn(obs.q_ex)
    ex_a = vect_fn(obs.a_ex)
    ex_v = vect_fn(obs.v_ex)
    ex_buses = np.zeros(obs.n_line)
    for line_id in range(obs.n_line):
        ex_buses[line_id] = topo[lex_pos[line_id]] * 1.0
    ex_bus = vect_fn(ex_buses)
    ex_rho = vect_fn(obs.rho)

    res = np.concatenate([
        # Time
        #time_v,
        time_line_cd,
        time_line_nm, time_line_dm,
        time_line_overflow,
        time_sub_cd,
        # Gens
        g_p, g_q, g_bus, g_v, #g_ar, g_tr,
        # Loads
        l_p, l_q, l_bus, l_v,
        # Origins
        or_p, or_q, or_a, or_bus, or_rho, or_v,
        # Extremities
        ex_p, ex_q, ex_a, ex_bus, ex_rho, ex_v
    ])
    return res + bias
