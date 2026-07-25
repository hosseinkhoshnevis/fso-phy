"""fsophy: baseband PHY simulation of an SDA-OCT-class free-space
optical link modem."""

from .config import SimConfig
from .sim import run_link

__version__ = "0.1.0"
__all__ = ["SimConfig", "run_link"]
