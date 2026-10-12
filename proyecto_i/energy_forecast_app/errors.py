# Módulo para describir errores de negocio

class DataLoadError(Exception):
    '''Error al cargar datos de origen'''

class ValidationError(Exception):
    '''Error de validación de los datos'''

class SchemaError(ValidationError):
    '''Faltan columnas obligatorias en los datos de origen'''
