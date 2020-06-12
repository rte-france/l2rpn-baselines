import os
import json

class SliceRDQN_Config():
    INITIAL_EPSILON = 0.80
    FINAL_EPSILON = 0.01
    DECAY_EPSILON = 1024*256
    STEP_EPSILON = (INITIAL_EPSILON-FINAL_EPSILON)/DECAY_EPSILON
    DISCOUNT_FACTOR = 0.99
    REPLAY_BUFFER_SIZE = 1024*8
    UPDATE_FREQ = 512
    UPDATE_TARGET_HARD_FREQ = -1
    UPDATE_TARGET_SOFT_TAU = 0.001
    INPUT_BIAS = 3.0
    SUFFLE_FREQ = 1000
    TRACE_LENGTH = 8
    BATCH_SIZE = 32
    LR = 1e-5
    VERBOSE = True

    @staticmethod
    def from_json(json_in_path):
        with open(json_in_path, 'r') as fp:
            conf_json = json.load(fp)
        
        for k,v in conf_json.items():
            if hasattr(DoubleDuelingDQNConfig, k):
                setattr(DoubleDuelingDQNConfig, k, v)

    @staticmethod
    def to_json(json_out_path):
        conf_json = {}
        for attr in dir(DoubleDuelingDQNConfig):
            if attr.startswith('__') or callable(attr):
                continue
            conf_json[attr] = getattr(DoubleDuelingDQNConfig, attr)

        with open(json_out_path, 'w+') as fp:
            json.dump(fp, conf_json, indent=2)
