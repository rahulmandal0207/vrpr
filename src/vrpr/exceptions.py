"""Exceptions raised by VRPR. Everything the CLI reports derives from VrprError."""


class VrprError(Exception):
    """Base class for expected, user-facing failures."""


class ConfigError(VrprError):
    """The configuration is missing, malformed or out of range."""


class DependencyError(VrprError):
    """An optional dependency needed for this operation is not installed."""


class ModelError(VrprError):
    """A model could not be found or loaded."""


class SourceError(VrprError):
    """A video source could not be opened."""


class OcrError(VrprError):
    """OCR failed on a single image. The pipeline logs this and carries on."""


class DisplayError(VrprError):
    """The live preview window cannot be shown in this environment."""
