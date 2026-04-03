try:
    from .utils import *
except ImportError:
    import warnings
    warnings.warn("src.utils import failed — transformers may not be installed. Utils that need transformers will fail.")
    from .utils import *  # still expose everything
