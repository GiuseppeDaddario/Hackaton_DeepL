from .act import * # noqa
from .config import * # noqa
from .encoder import * # noqa
from .head import * # noqa
from .layer import * # noqa
from .loader import * # noqa
from .loss import * # noqa
from .network import * # noqa
from .optimizer import * # noqa
from .pooling import * # noqa
from .stage import * # noqa
from .train import * # noqa
from .transform import * # noqa

from torch_geometric.graphgym.register import register_dataset
from .dataset.MyPPA6Classes import MyPPA6Classes

register_dataset('MyLocalPPA_A', MyPPA6Classes)
register_dataset('MyLocalPPA_B', MyPPA6Classes)
register_dataset('MyLocalPPA_C', MyPPA6Classes)
register_dataset('MyLocalPPA_D', MyPPA6Classes)
print("INFO: Custom dataset 'MyLocalPPA' registered successfully.")
