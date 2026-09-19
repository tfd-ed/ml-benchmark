"""Shared infrastructure for the ml-hardware-benchmark suite."""
import os

# PYTORCH_ENABLE_MPS_FALLBACK makes unsupported MPS operators run on the CPU
# *silently*. That would hide exactly what this suite is meant to expose, so it
# is removed from this process before torch is imported and the original value
# is kept for the run metadata. Without it, unsupported operators raise
# NotImplementedError and are recorded as status="unsupported".
MPS_FALLBACK_ENV_ORIGINAL = os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)

__version__ = "0.1.0"
