REGISTRY = {}

from .basic_controller import BasicMAC
from .continuous_controller import ContinuousMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["continuous_mac"] = ContinuousMAC
