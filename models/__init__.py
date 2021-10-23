from .dla_models import *
from .common import *
from .ggcnn import *
from .ggcnn2 import *
from .ggcnn3 import GGCNN3


def get_network(network_name):
    network_name = network_name.lower()
    if network_name == 'ggcnn':
        from .ggcnn import GGCNN
        return GGCNN
    elif network_name == 'ggcnn2':
        from .ggcnn2 import GGCNN2
        return GGCNN2
    elif network_name == 'ggcnn3':
        return GGCNN3
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))
