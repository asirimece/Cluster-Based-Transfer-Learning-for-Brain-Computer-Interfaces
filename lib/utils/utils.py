from omegaconf import OmegaConf
import pickle

def to_float(x, default):
    try:
        return float(OmegaConf.to_container(x, resolve=True))
    except Exception:
        return default
    
def _prefix_mtl_keys(state_dict):
    new_sd = {}
    for key, val in state_dict.items():
        if key.startswith("shared_backbone") or key.startswith("heads"):
            new_sd[f"mtl_net.{key}"] = val
        else:
            new_sd[key] = val
    return new_sd

def convert_state_dict_keys(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("shared_backbone") or key.startswith("heads"):
            new_key = "mtl_net." + key
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict

def remove_prefix(state_dict, prefix="mtl_net."):
    return {
        k[len(prefix):] if k.startswith(prefix) else k: v
        for k, v in state_dict.items()
    }
