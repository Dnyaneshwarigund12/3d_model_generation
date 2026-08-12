"""Photo -> 3D model + real-world measurements.

Import-light on purpose: the heavy modules (torch, rembg, gradio) are only
imported when the corresponding stage actually runs, so the geometry and scale
math can be tested on a machine with no GPU and no ML dependencies installed.
"""

__version__ = "0.1.0"
