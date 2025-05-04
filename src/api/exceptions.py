class BaseError(Exception):
    """Base class for all exceptions in this module."""

    def __init__(self, message: str, code: int):
        super().__init__(message)
        self.message = message
        self.code = code
        super().__init__(self.message)


class NotFoundError(BaseError):
    """Exception raised when a resource is not found."""

    def __init__(self, message: str, code: int = 404):
        super().__init__(message, code)
        self.message = message
        self.code = code


class GlassesDisconnectedError(BaseError):
    """Exception raised when glasses are disconnected."""

    def __init__(self, message: str, code: int = 503):
        super().__init__(message, code)
        self.message = message
        self.code = code


class RecordingAlreadyExistsError(BaseError):
    """Exception raised when a recording already exists."""

    def __init__(self, message: str, code: int = 409):
        super().__init__(message, code)
        self.message = message
        self.code = code


class InternalError(BaseError):
    """Exception raised when a runtime error occurs."""

    def __init__(self, message: str, code: int = 500):
        super().__init__(message, code)
        self.message = message
        self.code = code


# Labeler errors
class NoClassSelectedError(BaseError):
    """Exception raised when no class is selected."""

    message: str = "No class selected"
    code: int = 400

    def __init__(self) -> None:
        super().__init__(self.message, self.code)


class PredictionFailedError(BaseError):
    """Exception raised when prediction fails."""

    def __init__(self, message: str, code: int = 500):
        super().__init__(message, code)
        self.message = message
        self.code = code


class LabelingServiceNotAvailableError(BaseError):
    """Exception raised when trying to access an unloaded labeling service."""

    message: str = "Labeling service not available"
    code: int = 400

    def __init__(self) -> None:
        super().__init__(self.message, self.code)


class ImageEncodingError(BaseError):
    """Exception raised when image encoding fails."""

    def __init__(self, message: str, code: int = 500):
        super().__init__(message, code)
        self.message = message
        self.code = code


class TrackingJobAlreadyRunningError(BaseError):
    """Exception raised when a tracking job is already running."""

    message: str = "Tracking job already running"
    code: int = 400

    def __init__(self) -> None:
        super().__init__(self.message, self.code)
