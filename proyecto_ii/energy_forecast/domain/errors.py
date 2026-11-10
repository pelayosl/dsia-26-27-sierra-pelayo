"""Domain exceptions of the forecasting system."""


class DataLoadError(Exception):
    '''The source data could not be loaded'''


class ValidationError(Exception):
    '''The data does not meet the input contract'''


class SchemaError(ValidationError):
    '''Required columns are missing from the source data'''


class InsufficientHistoryError(Exception):
    '''There is not enough history before the origin to compute lags and rolling means'''


class ModelNotFoundError(Exception):
    '''There is no trained model to load'''


class ExternalServiceError(Exception):
    '''An external service failed (Open-Meteo, Hugging Face)'''
