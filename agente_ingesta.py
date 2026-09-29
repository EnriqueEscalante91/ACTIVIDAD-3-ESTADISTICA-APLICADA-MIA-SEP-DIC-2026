from fuentes_datos import load_datasets


def ingestar_datasets() -> str:
    """Punto de entrada del agente de ingesta para el orquestador local.

    Delega lectura, seleccion de proveedor y serializacion a fuentes_datos; asi
    este agente no contiene reglas dependientes de MySQL, CSV o Excel.
    """
    return load_datasets()