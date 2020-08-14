import warnings
import numpy as np
import math
import tensorflow as tf

def sac_size_obs(observation_space):
    dims = np.array([
        # Time
        5, # Timestamp
        4 * observation_space.n_line, # lines cd
        observation_space.n_sub, # Sub cd
        # Gen
        observation_space.n_gen * 6,
        # Load
        observation_space.n_load * 4,
        # Line origins
        observation_space.n_line * 4,
        # Line extremities
        observation_space.n_line * 5
    ])
    return np.sum(dims)

def to_norm_vect(inputv, pad_v = 0.0, scale_v = 1.0):
    v = np.asarray(inputv).astype(np.float32)
    v = v / scale_v
    vsafe = np.nan_to_num(v, nan=pad_v, posinf=pad_v, neginf=pad_v)
    return vsafe.astype(np.float32)

def sac_convert_obs(obs, bias=0.0):
    # Store some shortcuts
    topo = obs.topo_vect
    g_pos = obs.gen_pos_topo_vect
    l_pos = obs.load_pos_topo_vect
    lor_pos = obs.line_or_pos_topo_vect
    lex_pos = obs.line_ex_pos_topo_vect

    # Get time data
    time_li = [obs.month / 12.0, obs.day / 31.0, obs.day_of_week / 7.0,
               obs.hour_of_day / 24.0, obs.minute_of_hour / 60.0]
    time_v = to_norm_vect(time_li)
    time_line_cd = to_norm_vect(obs.time_before_cooldown_line,
                                pad_v=0.0, scale_v=12.0)
    time_line_nm = to_norm_vect(obs.time_next_maintenance,
                                scale_v=12.0*31.0*24.0)
    time_line_dm = to_norm_vect(obs.duration_next_maintenance,
                                scale_v=3.0*12.0*24.0)
    time_line_overflow = to_norm_vect(obs.timestep_overflow, scale_v=3.0)
    time_sub_cd = to_norm_vect(obs.time_before_cooldown_sub,
                               pad_v=0.0, scale_v=12.0)

    # Get generators info
    g_p = to_norm_vect(obs.prod_p, scale_v=500.0)
    g_q = to_norm_vect(obs.prod_q, scale_v=500.0)
    g_v = to_norm_vect(obs.prod_v, scale_v=250.0)
    g_tr = to_norm_vect(obs.target_dispatch, scale_v=500.0)
    g_ar = to_norm_vect(obs.actual_dispatch, scale_v=500.0)
    g_cost = to_norm_vect(obs.gen_cost_per_MW, pad_v=0.0, scale_v=100.0)
    g_buses = np.zeros(obs.n_gen)
    for gen_id in range(obs.n_gen):
        g_buses[gen_id] = topo[g_pos[gen_id]] * 1.0
    g_bus = to_norm_vect(g_buses, pad_v=0.0, scale_v=2.0)

    # Get loads info
    l_p = to_norm_vect(obs.load_p, scale_v=500.0)
    l_q = to_norm_vect(obs.load_q, scale_v=500.0)
    l_v = to_norm_vect(obs.load_v, scale_v=250.0)
    l_buses = np.zeros(obs.n_load)
    for load_id in range(obs.n_load):
        l_buses[load_id] = topo[l_pos[load_id]] * 1.0
    l_bus = to_norm_vect(l_buses, pad_v=0.0, scale_v=2.0)

    # Get lines origin info
    or_p = to_norm_vect(obs.p_or, scale_v=500.0)
    or_q = to_norm_vect(obs.q_or, scale_v=500.0)
    or_v = to_norm_vect(obs.v_or, scale_v=250.0)
    or_buses = np.zeros(obs.n_line)
    for line_id in range(obs.n_line):
        or_buses[line_id] = topo[lor_pos[line_id]] * 1.0
    or_bus = to_norm_vect(or_buses, pad_v=0.0, scale_v=2.0)
    or_rho = to_norm_vect(obs.rho, pad_v=0.0, scale_v=2.0)
    
    # Get extremities origin info
    ex_p = to_norm_vect(obs.p_ex, scale_v=500.0)
    ex_q = to_norm_vect(obs.q_ex, scale_v=500.0)
    ex_v = to_norm_vect(obs.v_ex, scale_v=250.0)
    ex_buses = np.zeros(obs.n_line)
    for line_id in range(obs.n_line):
        ex_buses[line_id] = topo[lex_pos[line_id]] * 1.0
    ex_bus = to_norm_vect(ex_buses, pad_v=0.0, scale_v=2.0)
    ex_rho = to_norm_vect(obs.rho, pad_v=0.0, scale_v=2.0)

    res = np.concatenate([
        # Time
        time_v,
        time_line_cd,
        time_line_nm, time_line_dm,
        time_line_overflow,
        time_sub_cd,
        # Gens
        g_p, g_q, g_v, g_ar, g_tr, g_bus,
        # Loads
        l_p, l_q, l_v, l_bus,
        # Origins
        or_p, or_q, or_v, or_bus,
        # Extremities
        ex_p, ex_q, ex_v, ex_bus, ex_rho
    ])
    return res + bias
