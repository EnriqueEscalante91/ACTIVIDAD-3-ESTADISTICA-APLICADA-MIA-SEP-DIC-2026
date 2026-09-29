import argparse
import json

from agente_ingesta import ingestar_datasets
from agente_limpieza import limpiar_datasets
from agente_planificador import planificar_produccion
from agente_reporte import generar_reporte


def ejecutar_pipeline(fecha_planificacion: str = "", algoritmo: str = "") -> dict:
    """Ejecuta los cuatro agentes locales y devuelve sus resultados principales.

    Las etapas se comunican mediante JSON para conservar un contrato sencillo e
    independiente del origen de datos; no se necesita servidor MCP ni red.
    """
    # Primera etapa: carga el conjunto base y todas las entidades adicionales
    # declaradas mediante configuracion de entorno.
    resultado_ingesta = ingestar_datasets()
    if json.loads(resultado_ingesta).get("status") != "success":
        # Detiene el flujo al fallar la fuente, sin pasar datos parciales a UCS.
        return {"etapa": "ingesta", "resultado": json.loads(resultado_ingesta)}

    # Segunda etapa: aplica reglas de calidad y deriva caracteristicas. La misma
    # fecha de referencia se pasa a limpieza y planificacion para ser consistente.
    resultado_limpieza = limpiar_datasets(resultado_ingesta, fecha_planificacion)
    datos_limpieza = json.loads(resultado_limpieza)
    if datos_limpieza.get("status") != "success":
        return {"etapa": "limpieza", "resultado": json.loads(resultado_limpieza)}

    # El despachador elige el motor por parametro o configuracion. El pipeline
    # solo conoce el contrato comun, no los detalles internos de cada algoritmo.
    resultado_plan = planificar_produccion(
        resultado_limpieza, fecha_planificacion, algoritmo
    )
    # El reporte presenta el resultado del motor elegido sin recalcular el plan.
    resultado_reporte = generar_reporte(resultado_plan)
    return {
        "etapa": "completada",
        "resumen_limpieza": datos_limpieza.get("cleaning_messages", []),
        "plan": json.loads(resultado_plan),
        "reporte": json.loads(resultado_reporte),
    }


def main() -> None:
    # Permite fijar la fecha desde consola para reproducir una misma corrida; si
    # se omite, cada agente utiliza la fecha actual del sistema.
    parser = argparse.ArgumentParser(description="Planeación de producción con algoritmo de búsqueda configurable")
    parser.add_argument(
        "--fecha-planificacion",
        default="",
        help="Fecha base en formato YYYY-MM-DD; por defecto se usa hoy.",
    )
    parser.add_argument(
        "--algoritmo",
        default="",
        help="Estrategia registrada a utilizar; por defecto SEARCH_ALGORITHM o ucs.",
    )
    arguments = parser.parse_args()
    result = ejecutar_pipeline(arguments.fecha_planificacion, arguments.algoritmo)
    cleaning_messages = result.get("resumen_limpieza", [])
    if cleaning_messages:
        print("RESUMEN DE LIMPIEZA DE DATOS")
        print("\n".join(cleaning_messages))
        print()
    report = result.get("reporte", {}).get("reporte")
    if report:
        # En caso de plan generado, prioriza mostrar texto legible al operador.
        print(report)
    else:
        # Si alguna etapa falla, imprime el objeto completo con contexto de etapa.
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # Solo ejecuta la interfaz CLI al invocar `python main.py`; importar main
    # desde pruebas u otro modulo no dispara el pipeline automaticamente.
    main()