from .nyud_v2 import NYUD
from .sun_rgbd import SUNRGBD

datasets = {
    'nyud': NYUD,
    'sunrgbd': SUNRGBD,
}

def get_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)