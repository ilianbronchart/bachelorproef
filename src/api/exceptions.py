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


class RuntimeError(BaseError):
    """Exception raised when a runtime error occurs."""

    def __init__(self, message: str, code: int = 500):
        super().__init__(message, code)
        self.message = message
        self.code = code
