import sys

from .hnet import models
from .hnet import modules

# Register submodules in sys.modules so that imports like
# "from hnet.models import ..." work correctly
sys.modules['hnet.models'] = models
sys.modules['hnet.modules'] = modules

from .hnet.models import *
from .hnet.modules import *
from .hnet.modules.utils import *