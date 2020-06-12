import os
import json

class DoubleDuelingDQNConfig():
    """
    DoubleDuelingDQN configurable hyperparameters
    exposed as class attributes
    """

    LR_DECAY_STEPS = 1024*64
    LR_DECAY_RATE = 0.95
    INITIAL_EPSILON = 0.99
    FINAL_EPSILON = 0.001
    DECAY_EPSILON = 1024*64
    DISCOUNT_FACTOR = 0.98
    PER_CAPACITY = 1024*64
    PER_ALPHA = 0.7
    PER_BETA = 0.5
    UPDATE_FREQ = 28
    UPDATE_TARGET_HARD_FREQ = -1
    UPDATE_TARGET_SOFT_TAU = 1e-3
    N_FRAMES = 4
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
