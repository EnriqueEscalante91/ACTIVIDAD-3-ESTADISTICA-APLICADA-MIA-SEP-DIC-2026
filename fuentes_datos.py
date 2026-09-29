import json
import os
import re
from pathlib import Path

import polars as pl


# Estos tres datasets forman el contrato minimo que requiere el planificador.
# Cada entrada contiene: variable de ruta de archivo, nombre base/tabla y
# variable opcional para sobrescribir el nombre de tabla MySQL.
CORE_DATASETS = {
    "orders": (
        "ERP_ORDERS_FILE", "Pedidos_Pendientes", "pedidos_pendientes", "MYSQL_TABLE_ORDERS"
    ),
    "inventory": (
        "ERP_INVENTORY_FILE", "Inventarios_Almacen", "inventarios_almacen", "MYSQL_TABLE_INVENTORY"
    ),
    "capacity": (
        "ERP_CAPACITY_FILE", "Capacidad_Operativa", "capacidad_operativa", "MYSQL_TABLE_CAPACITY"
    ),
}


def _read_file(dataset: str, file_env: str, default_name: str) -> pl.DataFrame:
    # Las rutas relativas se resuelven respecto a este proyecto, no respecto al
    # directorio desde el que el usuario haya lanzado main.py.
    path = Path(os.getenv(file_env, f"{default_name}.csv"))
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    if not path.is_file():
        raise FileNotFoundError(f"No existe el archivo configurado para {dataset}: {path}")

    # El formato se determina por extension. Polars interpreta fechas del CSV
    # cuando puede; en Excel se usa Calamine y opcionalmente una hoja configurada.
    extension = path.suffix.lower()
    if extension == ".csv":
        return pl.read_csv(path, try_parse_dates=True)
    if extension in {".xlsx", ".xlsm"}:
        sheet = os.getenv(f"ERP_{dataset.upper()}_SHEET")
        return pl.read_excel(path, engine="calamine", sheet_name=sheet) if sheet else pl.read_excel(
            path, engine="calamine"
        )
    raise ValueError(f"Formato no compatible para {dataset}: usa CSV, XLSX o XLSM.")


def _read_mysql(dataset: str, table_env: str, default_table: str) -> pl.DataFrame:
    # Import local: pymysql solo se necesita cuando DATA_SOURCE=mysql, por lo que
    # los usuarios que trabajan con archivos no requieren abrir una conexion.
    import pymysql

    table = os.getenv(table_env, default_table)
    # Los nombres de tabla SQL no se pueden parametrizar como valores. Esta
    # validacion estricta impide insertar fragmentos SQL desde una variable env.
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
        raise ValueError(f"Nombre de tabla no válido: {table}")

    # Credenciales y ubicacion se leen del entorno; no se guardan secretos en
    # codigo. Los timeouts evitan dejar el proceso bloqueado indefinidamente.
    connection = pymysql.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", ""),
        database=os.getenv("MYSQL_DATABASE", "erp_pro"),
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=10,
        read_timeout=60,
    )
    try:
        # DictCursor entrega filas como diccionarios. El identificador de tabla
        # ya validado se delimita con backticks para soportar nombres SQL validos.
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT * FROM `{table}`")
            rows = cursor.fetchall()
            columns = [column[0] for column in cursor.description or []]
        return pl.DataFrame(rows, schema=columns, orient="row") if rows else pl.DataFrame(
            schema=columns
        )
    finally:
        # La conexion se cierra incluso si la consulta o la conversion falla.
        connection.close()


def _configured_datasets() -> dict[str, tuple[str, str, str, str]]:
    # Copia el registro base para no mutarlo globalmente entre llamadas.
    datasets = dict(CORE_DATASETS)
    # ERP_EXTRA_DATASETS acepta nombres separados por coma, por ejemplo
    # "facturas,notas_credito,entregas". Para cada uno se derivan sus variables
    # de archivo y tabla sin agregar codigo especifico por tipo documental.
    extra_names = os.getenv("ERP_EXTRA_DATASETS", "").split(",")
    for raw_name in extra_names:
        name = raw_name.strip().lower()
        if not name:
            continue
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError(f"Nombre de dataset no válido: {raw_name}")
        if name in datasets:
            raise ValueError(f"El dataset {name} ya está incluido como obligatorio.")
        suffix = name.upper()
        # El nombre normalizado sirve como nombre por defecto tanto del CSV
        # (nombre.csv) como de la tabla MySQL; las variables permiten cambiarlos.
        datasets[name] = (f"ERP_{suffix}_FILE", f"{name}.csv", name, f"MYSQL_TABLE_{suffix}")
    return datasets


def load_datasets() -> str:
    """Lee datasets base y extensiones configuradas desde MySQL o CSV/Excel."""
    # El modo determina una fuente uniforme para todos los datasets de esta
    # ejecucion: archivos o MySQL. No se mezclan proveedores implicitamente.
    source = os.getenv("DATA_SOURCE", "files").strip().lower()
    if source not in {"files", "mysql"}:
        return json.dumps({"status": "error", "error": "DATA_SOURCE debe ser 'files' o 'mysql'."})

    try:
        datasets = {}
        registry = _configured_datasets()
        # Recorre el registro en orden y serializa cada tabla como lista de filas
        # JSON. Ese contrato comun permite que los siguientes agentes no sepan
        # si los datos originales venian de un archivo o de una base de datos.
        for dataset, (file_env, default_file, table_name, table_env) in registry.items():
            if source == "mysql":
                frame = _read_mysql(dataset, table_env, table_name)
            else:
                frame = _read_file(dataset, file_env, default_file)
            datasets[dataset] = json.loads(frame.write_json())
        return json.dumps({
            "status": "success",
            "source": source,
            "required_datasets": list(CORE_DATASETS),
            "datasets": datasets,
        })
    except Exception as error:
        # Se devuelve el error en el mismo formato JSON que una respuesta normal,
        # para que main.py pueda detener el pipeline y mostrar la etapa fallida.
        return json.dumps({"status": "error", "source": source, "error": str(error)})