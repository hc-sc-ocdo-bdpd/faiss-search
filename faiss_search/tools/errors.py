from file_processing.tools.errors import FileProcessorError

class FileTypeError(FileProcessorError):
    """Raised when the provided file is of the incorrect file type."""

class EncodingModelError(FileProcessorError):
    """Raised when there is no encoding model specified."""
