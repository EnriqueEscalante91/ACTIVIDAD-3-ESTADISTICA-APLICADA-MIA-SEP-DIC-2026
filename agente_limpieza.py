import json
import re
import unicodedata
from datetime import date, timedelta

import polars as pl

# Cada clave identifica un tipo de entidad que la planificacion sabe procesar.
# El diccionario interno traduce nombres de columna normalizados (sin acentos ni
# diferencias de mayusculas) al nombre canonico que esperan los demas agentes.
SCHEMAS = {
    "orders": {
        "id_pedido": "ID_Pedido",
        "producto": "Producto",
        "id_cliente": "ID_Cliente",
        "nombre_cliente": "Nombre_Cliente",
        "canal_comercial": "Canal_Comercial",
        "prioridad_cliente": "Prioridad_Cliente",
        "fecha_entrega_inicio": "Fecha_Entrega_Inicio",
        "fecha_entrega_fin": "Fecha_Entrega_Fin",
        "cantidad_solicitada": "Cantidad_Solicitada",
    },
    "inventory": {
        "id_ubicacion": "ID_Ubicacion",
        "producto": "Producto",
        "inventario_actual": "Inventario_Actual",
        "capacidad_maxima": "Capacidad_Maxima",
        "abastecimiento_minimo": "Abastecimiento_Minimo",
    },
    "capacity": {
        "id_linea": "ID_Linea",
        "producto_asignable": "Producto_Asignable",
        "horas_turno_disponibles": "Horas_Turno_Disponibles",
        "tasa_produccion_por_hora": "Tasa_Produccion_Por_Hora",
    },
}

# Columnas minimas que debe traer cada entidad base. Las columnas adicionales
# se conservan, pero estas son necesarias para validar y preparar el modelo UCS.
REQUIRED = {
    "orders": [
        "ID_Pedido", "Producto", "Canal_Comercial", "Prioridad_Cliente",
        "Fecha_Entrega_Inicio", "Fecha_Entrega_Fin", "Cantidad_Solicitada",
    ],
    "inventory": [
        "ID_Ubicacion", "Producto", "Inventario_Actual", "Capacidad_Maxima",
        "Abastecimiento_Minimo",
    ],
    "capacity": [
        "ID_Linea", "Producto_Asignable", "Horas_Turno_Disponibles",
        "Tasa_Produccion_Por_Hora",
    ],
}

# Claves de identidad usadas para quitar duplicados en los datasets conocidos.
# En pedidos basta el folio; inventario y capacidad identifican filas por pares,
# porque un producto puede repetirse en distintas ubicaciones o lineas.
KEYS = {
    "orders": ["ID_Pedido"],
    "inventory": ["ID_Ubicacion", "Producto"],
    "capacity": ["ID_Linea", "Producto_Asignable"],
}


def _normalize_name(value: str) -> str:
    # Hace comparables nombres como "Prioridad Cliente", "prioridad_cliente"
    # y "prioridad del cliente": elimina acentos y reemplaza separadores por '_'.
    decomposed = unicodedata.normalize("NFKD", value.strip().lower())
    without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "_", without_accents).strip("_")


def _canonicalize(frame: pl.DataFrame, aliases: dict[str, str]) -> pl.DataFrame:
    # Renombra solo columnas reconocidas por el esquema de la entidad. Las
    # columnas desconocidas no se eliminan para no perder informacion de origen.
    rename = {}
    for column in frame.columns:
        normalized = _normalize_name(column)
        canonical = aliases.get(normalized)
        if canonical and canonical not in frame.columns:
            rename[column] = canonical
    return frame.rename(rename) if rename else frame


def _category(value: object, aliases: dict[str, str]) -> str | None:
    # Convierte categorias conocidas mediante el catalogo de equivalencias.
    # Si aparece una categoria nueva, la conserva normalizada en mayusculas en
    # vez de asignarle silenciosamente una categoria de negocio incorrecta.
    if value is None:
        return None
    normalized = _normalize_name(str(value))
    return aliases.get(normalized, normalized.upper())


def _parse_date(column: str) -> pl.Expr:
    # Intenta tres formatos frecuentes de ERP. strict=False convierte fechas no
    # interpretables en nulos; el llamador decide despues si descarta esa fila.
    text = pl.col(column).cast(pl.String, strict=False)
    return pl.coalesce(
        text.str.strptime(pl.Date, "%Y-%m-%d", strict=False),
        text.str.strptime(pl.Date, "%Y/%m/%d", strict=False),
        text.str.strptime(pl.Date, "%d/%m/%Y", strict=False),
        text.str.strptime(pl.Date, "%Y-%m-%d %H:%M:%S", strict=False),
    )


def _fill_numeric(frame: pl.DataFrame, columns: list[str]) -> tuple[pl.DataFrame, dict[str, int]]:
    # Convierte las columnas configuradas a Float64 y reemplaza nulos (incluidos
    # textos numericos invalidos convertidos a null) por la mediana de la columna.
    # Si toda la columna esta vacia, usa cero como valor de respaldo.
    expressions = []
    imputed = {}
    for column in columns:
        if column not in frame.columns:
            continue
        numeric = pl.col(column).cast(pl.Float64, strict=False)
        median = frame.select(numeric.median()).item()
        imputed[column] = int(frame.select(numeric.is_null().sum()).item() or 0)
        expressions.append(numeric.fill_null(median if median is not None else 0).alias(column))
    return (frame.with_columns(expressions) if expressions else frame), imputed


def _add_outlier_flags(frame: pl.DataFrame, columns: list[str]) -> tuple[pl.DataFrame, dict[str, int]]:
    # Marca posibles valores atipicos con el criterio IQR: fuera de
    # [Q1 - 1.5*IQR, Q3 + 1.5*IQR]. No los borra ni los corrige, para que una
    # observacion real extrema no desaparezca automaticamente del analisis.
    expressions = []
    thresholds = {}
    for column in columns:
        values = frame[column].drop_nulls()
        # Con menos de cuatro observaciones el rango intercuartil no es una base
        # suficientemente util para crear una marca estadistica.
        if len(values) < 4:
            continue
        lower_quartile = values.quantile(0.25)
        upper_quartile = values.quantile(0.75)
        if lower_quartile is None or upper_quartile is None:
            continue
        spread = upper_quartile - lower_quartile
        low = lower_quartile - 1.5 * spread
        high = upper_quartile + 1.5 * spread
        expressions.append((~pl.col(column).is_between(low, high, closed="both")).alias(
            f"Es_Outlier_{column}"
        ))
        thresholds[column] = (low, high)
    if not expressions:
        return frame, {}
    frame = frame.with_columns(expressions)
    flagged = {
        column: int(frame.select(pl.col(f"Es_Outlier_{column}").sum()).item() or 0)
        for column in thresholds
    }
    return frame, flagged


def _scale_columns(frame: pl.DataFrame, columns: list[str]) -> tuple[pl.DataFrame, list[str]]:
    # Agrega una version estandarizada (z-score) cuando hay variacion. La columna
    # original se conserva; si la desviacion es cero o no calculable, no se crea
    # una columna escalada porque dividir por cero produciria valores invalidos.
    expressions = []
    scaled = []
    for column in columns:
        if column not in frame.columns:
            continue
        mean = frame[column].mean()
        deviation = frame[column].std()
        if mean is not None and deviation is not None and deviation > 0:
            scaled.append(f"{column}_Escala_Z")
            expressions.append(((pl.col(column) - mean) / deviation).alias(scaled[-1]))
    return (frame.with_columns(expressions) if expressions else frame), scaled


def _normalize_category_column(
    frame: pl.DataFrame, column: str, aliases: dict[str, str]
) -> tuple[pl.DataFrame, int]:
    normalized = pl.col(column).map_elements(
        lambda value: _category(value, aliases), return_dtype=pl.String
    )
    changed = int(
        frame.select((normalized != pl.col(column)).fill_null(False).sum()).item() or 0
    )
    return frame.with_columns(normalized.alias(column)), changed


def _build_cleaning_messages(summary: dict[str, dict]) -> list[str]:
    """Convierte las métricas detalladas en mensajes breves para consola."""
    dataset_labels = {
        "orders": "Pedidos",
        "inventory": "Inventario",
        "capacity": "Capacidad operativa",
    }
    rule_labels = {
        "fechas_o_producto_faltantes_invalidos": "fecha o producto faltante/inválido",
        "cantidad_no_positiva": "cantidad no positiva",
        "inventario_negativo": "inventario negativo",
        "capacidad_o_tasa_invalida": "horas o tasa de producción inválidas",
    }
    messages = []
    for dataset, metrics in summary.items():
        messages.append(
            f"{dataset_labels.get(dataset, dataset)}: {metrics['filas_entrada']} registros de entrada, "
            f"{metrics['filas_salida']} de salida; "
            f"{metrics['filas_eliminadas_total']} eliminados en total."
        )
        if metrics.get("datos_modificados") is False:
            messages.append(f"  - Sin reglas de limpieza aplicadas; datos preservados intactos.")
            continue

        messages.append(f"  - Duplicados eliminados: {metrics['duplicados_eliminados']}.")
        messages.append(
            f"  - Filas sin clave eliminadas: {metrics['filas_sin_clave_eliminadas']}."
        )
        renamed = metrics["columnas_renombradas"]
        messages.append(
            f"  - Columnas estandarizadas: {renamed if renamed else '0; encabezados ya canónicos'}."
        )
        for column, amount in metrics["valores_numericos_imputados"].items():
            messages.append(f"  - Valores numéricos imputados en {column}: {amount}.")
        for column, amount in metrics["valores_categoria_normalizados"].items():
            messages.append(f"  - Categorías normalizadas en {column}: {amount}.")
        for column, amount in metrics["espacios_textuales_recortados"].items():
            if amount:
                messages.append(f"  - Textos con espacios externos corregidos en {column}: {amount}.")
        for column, amount in metrics["valores_fecha_faltantes"].items():
            messages.append(f"  - Fechas faltantes en {column}: {amount}.")
        for column, amount in metrics["valores_fecha_invalidos"].items():
            messages.append(f"  - Fechas inválidas en {column}: {amount}.")
        for rule, amount in metrics["filas_eliminadas_por_reglas"].items():
            messages.append(
                f"  - Filas descartadas por {rule_labels.get(rule, rule)}: {amount}."
            )
        for column, amount in metrics["outliers_marcados"].items():
            messages.append(f"  - Outliers marcados en {column}: {amount}.")
        derived = metrics.get("caracteristicas_derivadas", [])
        if derived:
            messages.append(f"  - Características calculadas: {', '.join(derived)}.")
        messages.append(
            f"  - Columnas escaladas (z-score): "
            f"{', '.join(metrics['columnas_escaladas']) or 'ninguna'}."
        )
    return messages


def _clean_dataset(name: str, records: list[dict], reference_date: date) -> pl.DataFrame:
    # Esta ruta solo se usa para pedidos, inventario y capacidad, cuyos campos y
    # reglas estan definidos. Las entidades futuras pasan por _clean_extension.
    input_rows = len(records)
    frame = pl.DataFrame(records)
    renamed_columns = {}
    for column in frame.columns:
        canonical = SCHEMAS[name].get(_normalize_name(column))
        if canonical and canonical != column and canonical not in frame.columns:
            renamed_columns[column] = canonical
    frame = _canonicalize(frame, SCHEMAS[name])
    missing = sorted(set(REQUIRED[name]) - set(frame.columns))
    if missing:
        raise ValueError(f"{name}: faltan columnas requeridas: {', '.join(missing)}")

    # Descarta filas sin identificador y conserva la primera fila de cada clave.
    # La politica de "primera" es determinista respecto al orden recibido.
    rows_without_identity = frame.height - frame.drop_nulls(KEYS[name]).height
    frame = frame.drop_nulls(KEYS[name])
    rows_before_deduplication = frame.height
    frame = frame.unique(subset=KEYS[name], keep="first", maintain_order=True)
    duplicates_removed = rows_before_deduplication - frame.height

    # La imputacion numerica se limita a cantidades operativas declaradas por
    # entidad; no se imputan identificadores, fechas ni categorias.
    numeric_columns = {
        "orders": ["Cantidad_Solicitada"],
        "inventory": ["Inventario_Actual", "Capacidad_Maxima", "Abastecimiento_Minimo"],
        "capacity": ["Horas_Turno_Disponibles", "Tasa_Produccion_Por_Hora"],
    }[name]
    frame, imputed_by_column = _fill_numeric(frame, numeric_columns)

    # Quita espacios externos de todos los campos de texto conocidos. No cambia
    # mayusculas/minusculas de nombres de clientes o productos.
    string_columns = [column for column, dtype in frame.schema.items() if dtype == pl.String]
    trimmed_by_column = {
        column: int(
            frame.select((pl.col(column) != pl.col(column).str.strip_chars()).sum()).item() or 0
        )
        for column in string_columns
    }
    frame = frame.with_columns([pl.col(column).str.strip_chars() for column in string_columns])

    metrics = {
        "filas_entrada": input_rows,
        "columnas_renombradas": renamed_columns,
        "filas_sin_clave_eliminadas": rows_without_identity,
        "duplicados_eliminados": duplicates_removed,
        "valores_numericos_imputados": imputed_by_column,
        "espacios_textuales_recortados": trimmed_by_column,
        "valores_categoria_normalizados": {},
        "valores_fecha_faltantes": {},
        "valores_fecha_invalidos": {},
        "filas_eliminadas_por_reglas": {},
        "outliers_marcados": {},
        "columnas_escaladas": [],
        "caracteristicas_derivadas": [],
    }

    if name == "orders":
        metrics["caracteristicas_derivadas"] = [
            "Canal_Codigo", "Prioridad_Codigo", "Dias_Rezago",
            "Dias_Ventana_Entrega", "Urgencia_Logistica",
        ]
        # Los alias permiten unificar variantes de escritura documentadas. Una
        # categoria desconocida se conserva como texto normalizado y recibe
        # codigo 0 mas abajo, para hacerla visible sin atribuirle prioridad.
        channel_aliases = {
            "EXPORT": "EXPORTACION", "EXPORTACION": "EXPORTACION",
            "AUTOSERVICIO": "AUTOSERVICIO", "AUTO_SERVICIO": "AUTOSERVICIO",
            "MAYORISTA": "MAYORISTA", "GRANEL": "GRANEL",
        }
        priority_aliases = {"ALTA": "ALTA", "MEDIA": "MEDIA", "BAJA": "BAJA"}
        frame, metrics["valores_categoria_normalizados"]["Canal_Comercial"] = (
            _normalize_category_column(frame, "Canal_Comercial", channel_aliases)
        )
        frame, metrics["valores_categoria_normalizados"]["Prioridad_Cliente"] = (
            _normalize_category_column(frame, "Prioridad_Cliente", priority_aliases)
        )
        parsed_dates = {
            "Fecha_Entrega_Inicio": _parse_date("Fecha_Entrega_Inicio"),
            "Fecha_Entrega_Fin": _parse_date("Fecha_Entrega_Fin"),
        }
        for column, parsed in parsed_dates.items():
            parsed_values = frame.select(parsed.alias(column)).to_series()
            source_nulls = int(frame.select(pl.col(column).is_null().sum()).item() or 0)
            invalid_dates = int(parsed_values.is_null().sum()) - source_nulls
            metrics["valores_fecha_faltantes"][column] = source_nulls
            metrics["valores_fecha_invalidos"][column] = max(0, invalid_dates)
        frame_before_date_filter = frame.height
        frame = frame.with_columns(
            parsed_dates["Fecha_Entrega_Inicio"].alias("Fecha_Entrega_Inicio"),
            parsed_dates["Fecha_Entrega_Fin"].alias("Fecha_Entrega_Fin"),
        ).drop_nulls(["Fecha_Entrega_Inicio", "Fecha_Entrega_Fin", "Producto"])
        metrics["filas_eliminadas_por_reglas"]["fechas_o_producto_faltantes_invalidos"] = (
            frame_before_date_filter - frame.height
        )
        # Sin ventana logistica valida, producto o cantidad positiva, el pedido
        # no puede entrar al planificador y se excluye de la salida curada.
        frame_before_quantity_filter = frame.height
        frame = frame.filter(pl.col("Cantidad_Solicitada") > 0)
        metrics["filas_eliminadas_por_reglas"]["cantidad_no_positiva"] = (
            frame_before_quantity_filter - frame.height
        )
        frame = frame.with_columns(
            # Los codigos son ordinales descriptivos, no costos monetarios. El
            # planificador aplica las penalizaciones con sus propias reglas.
            pl.col("Canal_Comercial").replace(
                {"EXPORTACION": 4, "AUTOSERVICIO": 3, "MAYORISTA": 2, "GRANEL": 1}, default=0
            ).cast(pl.Int8).alias("Canal_Codigo"),
            pl.col("Prioridad_Cliente").replace(
                {"ALTA": 3, "MEDIA": 2, "BAJA": 1}, default=0
            ).cast(pl.Int8).alias("Prioridad_Codigo"),
            (pl.lit(reference_date) - pl.col("Fecha_Entrega_Fin")).dt.total_days()
            .clip(lower_bound=0).alias("Dias_Rezago"),
            (pl.col("Fecha_Entrega_Fin") - pl.col("Fecha_Entrega_Inicio"))
            .dt.total_days().alias("Dias_Ventana_Entrega"),
        ).with_columns(
            # La urgencia se deriva unicamente de la fecha final: vencido si ya
            # paso, alta dentro de 3 dias, media dentro de 7, y normal despues.
            pl.when(pl.col("Dias_Rezago") > 0).then(pl.lit("VENCIDO"))
            .when(pl.col("Fecha_Entrega_Fin") <= pl.lit(reference_date + timedelta(days=3)))
            .then(pl.lit("ALTA"))
            .when(pl.col("Fecha_Entrega_Fin") <= pl.lit(reference_date + timedelta(days=7)))
            .then(pl.lit("MEDIA"))
            .otherwise(pl.lit("NORMAL")).alias("Urgencia_Logistica")
        )
        frame, metrics["outliers_marcados"] = _add_outlier_flags(frame, ["Cantidad_Solicitada"])
        frame, metrics["columnas_escaladas"] = _scale_columns(frame, ["Cantidad_Solicitada"])
    elif name == "inventory":
        # Un inventario negativo se trata como dato invalido. No se filtra por
        # capacidad o stock minimo aqui; esas condiciones pertenecen al modelo
        # operativo y deben permanecer disponibles para su evaluacion.
        rows_before_validation = frame.height
        frame = frame.filter(pl.col("Inventario_Actual") >= 0)
        metrics["filas_eliminadas_por_reglas"]["inventario_negativo"] = (
            rows_before_validation - frame.height
        )
        frame, metrics["outliers_marcados"] = _add_outlier_flags(
            frame, ["Inventario_Actual", "Capacidad_Maxima"]
        )
        frame, metrics["columnas_escaladas"] = _scale_columns(
            frame, ["Inventario_Actual", "Capacidad_Maxima"]
        )
    else:
        # Una linea no puede planificarse con horas negativas o tasa nula/negativa.
        # La tasa debe ser positiva para calcular horas = cantidad / tasa.
        rows_before_validation = frame.height
        frame = frame.filter(
            (pl.col("Horas_Turno_Disponibles") >= 0) & (pl.col("Tasa_Produccion_Por_Hora") > 0)
        )
        metrics["filas_eliminadas_por_reglas"]["capacidad_o_tasa_invalida"] = (
            rows_before_validation - frame.height
        )
        frame, metrics["outliers_marcados"] = _add_outlier_flags(
            frame, ["Horas_Turno_Disponibles", "Tasa_Produccion_Por_Hora"]
        )
        frame, metrics["columnas_escaladas"] = _scale_columns(
            frame, ["Horas_Turno_Disponibles", "Tasa_Produccion_Por_Hora"]
        )

    metrics["filas_salida"] = frame.height
    metrics["filas_eliminadas_total"] = input_rows - frame.height
    return frame, metrics


def _clean_extension(records: list[dict]) -> tuple[pl.DataFrame, dict]:
    # No se aplican transformaciones a entidades sin reglas de dominio registradas.
    # Por ejemplo, lineas repetidas en una factura pueden ser movimientos validos;
    # imputar o deduplicar aqui podria alterar importes o trazabilidad.
    frame = pl.DataFrame(records)
    return frame, {
        "registros_entrada": frame.height,
        "registros_salida": frame.height,
        "columnas": frame.columns,
        "reglas_especificas": False,
        "datos_modificados": False,
    }


def limpiar_datasets(datasets_json: str, fecha_referencia: str = "") -> str:
    """Limpia datasets conocidos y conserva entidades nuevas sin reglas asumidas.

    Recibe el JSON de ingesta y devuelve el mismo contrato general con los datos
    curados, perfiles de calidad y notas sobre las reglas que se aplicaron.
    """
    try:
        source = json.loads(datasets_json)
        if source.get("status") != "success":
            return json.dumps({"status": "error", "error": "La ingesta no devolvió datasets válidos."})
        # Se usa una fecha reproducible cuando se proporciona; si no, se toma el
        # dia local actual para calcular rezago y urgencia de los pedidos.
        reference = date.fromisoformat(fecha_referencia) if fecha_referencia else date.today()
        incoming = source.get("datasets", {})
        # Los tres datasets base son obligatorios mientras la planificacion UCS
        # dependa de pedidos, existencias y recursos de fabrica.
        missing_core = sorted(set(SCHEMAS) - set(incoming))
        if missing_core:
            return json.dumps({
                "status": "error",
                "error": "Faltan datasets requeridos para la planificación: " + ", ".join(missing_core),
            })

        cleaned = {}
        profiles = {}
        cleaning_summary = {}
        # Mantiene todas las entidades recibidas. Las conocidas pasan por reglas
        # de dominio; cualquier otra se conserva sin modificar y se perfila.
        for name, records in incoming.items():
            if name in SCHEMAS:
                frame, cleaning_summary[name] = _clean_dataset(name, records, reference)
                profiles[name] = {
                    "registros_salida": frame.height,
                    "columnas": frame.columns,
                    "reglas_especificas": True,
                }
            else:
                frame, profiles[name] = _clean_extension(records)
                cleaning_summary[name] = {
                    "filas_entrada": frame.height,
                    "filas_salida": frame.height,
                    "filas_eliminadas_total": 0,
                    "datos_modificados": False,
                }
            cleaned[name] = json.loads(frame.write_json())
        return json.dumps({
            "status": "success",
            "reference_date": reference.isoformat(),
            "datasets": cleaned,
            "dataset_profiles": profiles,
            "cleaning_summary": cleaning_summary,
            "cleaning_messages": _build_cleaning_messages(cleaning_summary),
            "quality_notes": [
                # Estas notas viajan con los datos para que los agentes siguientes
                # sepan que se marco, que no se calculo y que no debe suponerse.
                "Outliers se marcan para revisión y no se eliminan automáticamente.",
                "Datasets adicionales se transportan intactos y requieren reglas explícitas de entidad antes de limpiarse o afectar decisiones.",
                "Las reglas de planificación solo consumen pedidos, inventario y capacidad.",
            ],
        })
    except Exception as error:
        return json.dumps({"status": "error", "error": str(error)})
