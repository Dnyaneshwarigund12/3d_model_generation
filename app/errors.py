"""Exception types shared across pipeline stages.

Each stage raises a specific type so the UI can show the user something
actionable ("no marker found in the photo") instead of a stack trace.
"""

from __future__ import annotations


class PipelineError(RuntimeError):
    """Base class for every recoverable failure in the pipeline."""


class SegmentationError(PipelineError):
    """Background removal produced no usable subject."""


class ScaleError(PipelineError):
    """No real-world scale could be established from the inputs."""


class GeneratorError(PipelineError):
    """A 3D generation backend is unavailable or failed on this input."""


class MeasurementError(PipelineError):
    """The mesh could not be measured."""
