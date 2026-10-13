"""Excepciones de dominio del pipeline de ventas."""

class DataLoadError(Exception):
    '''Error al cargar datos de origen'''

class ValidationError(Exception):
    '''Error de validación de los datos'''