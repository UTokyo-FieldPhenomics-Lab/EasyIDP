""" All the custom exceptions types
"""


class PylasError(Exception):
    pass


class UnknownExtraType(PylasError):
    pass


class PointFormatNotSupported(PylasError):
    pass


class FileVersionNotSupported(PylasError):
    pass


class LazError(PylasError):
    pass


class IncompatibleDataFormat(PylasError):
    pass
