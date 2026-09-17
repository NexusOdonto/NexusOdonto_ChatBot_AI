"""Script de utilidad para purgar el Caché Semántico y restablecer sesiones en PostgreSQL.

Uso:
    python scripts/limpiar_cache_y_sesiones.py [--clear-threads] [--thread-id <NUMERO>]
"""

import sys
import os
import argparse
import asyncio
import logging

# Añadir el directorio raíz al path para importar módulos de la app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings
from app.services.semantic_cache import purgar_cache_semantico
from app.agents.tools.qdrant_tool import get_qdrant_client
from app.session.postgres_checkpointer import PostgresCheckpointer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LimpiarCache")


async def main():
    parser = argparse.ArgumentParser(description="Purga de Caché Semántico y sesiones de Nexus Odonto")
    parser.add_argument("--clear-all-threads", action="store_true", help="Elimina TODOS los checkpoints en PostgreSQL")
    parser.add_argument("--thread-id", type=str, default="", help="Elimina checkpoints de un thread_id específico")
    args = parser.parse_args()

    print("=" * 65)
    print("🧹 LIMPIEZA DE CACHÉ SEMÁNTICO Y SESIONES - NEXUS ODONTO")
    print("=" * 65)

    # 1. Purgar caché semántico en Qdrant
    print("\n[1/2] Purgando colección 'semantic_cache' en Qdrant...")
    try:
        exito = purgar_cache_semantico()
        if exito:
            print("  ✅ Colección 'semantic_cache' eliminada y recreada vacía.")
        else:
            print("  ⚠️ No se pudo purgar Qdrant (verifica que el contenedor/servicio esté activo).")
    except Exception as e:
        print(f"  ❌ Error al conectar con Qdrant: {e}")

    # 2. Limpieza de checkpoints en PostgreSQL
    print("\n[2/2] Gestión de sesiones en PostgreSQL...")
    if args.clear_all_threads or args.thread_id:
        try:
            checkpointer = PostgresCheckpointer(settings.postgres_checkpoint_url)
            await checkpointer.start()

            if args.thread_id:
                print(f"  🗑️ Limpiando checkpoints para thread_id: '{args.thread_id}'...")
                await checkpointer.clear_thread(args.thread_id)
                print(f"  ✅ Sesión '{args.thread_id}' eliminada con éxito.")

            if args.clear_all_threads:
                print("  🗑️ Limpiando TODOS los checkpoints en PostgreSQL...")
                async with checkpointer.pool.connection() as conn:
                    await conn.execute("TRUNCATE checkpoints, checkpoint_blobs, checkpoint_writes;")
                print("  ✅ Todos los checkpoints eliminados.")

            await checkpointer.stop()
        except Exception as e:
            print(f"  ⚠️ No se pudo conectar a PostgreSQL: {e}")
    else:
        print("  ℹ️ No se solicitaron limpiezas de PostgreSQL.")
        print("     Usa --thread-id <numero> o --clear-all-threads si deseas reiniciar historiales.")

    print("\n" + "=" * 65)
    print("🎉 Proceso finalizado.")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(main())
