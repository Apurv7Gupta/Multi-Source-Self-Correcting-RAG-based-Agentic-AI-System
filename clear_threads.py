import asyncio
import os
from dotenv import load_dotenv
from psycopg_pool import AsyncConnectionPool
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment variables from .env file
load_dotenv()

DB_USER = os.environ.get("PGSQL_USERNAME")
DB_PASSWORD = os.environ.get("PGSQL_PASSWORD")
DB_HOST = os.environ.get("PGSQL_HOST", "localhost")
DB_PORT = os.environ.get("PGSQL_PORT", "5432")
DB_NAME = os.environ.get("PGSQL_NAME")

DB_URI = (
    f"postgresql://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    f"?sslmode=require&channel_binding=require"
)

async def clear_threads():
    print("Connecting to database and deleting LangGraph memory (threads)...")
    try:
        async with AsyncConnectionPool(
            conninfo=DB_URI,
            max_size=1,
            min_size=1,
            kwargs={"autocommit": True},
        ) as pool:
            async with pool.connection() as conn:
                # Truncate checkpointer tables. 
                # LangGraph typically uses 'checkpoints', 'checkpoint_writes', 'checkpoint_blobs'.
                await conn.execute("""
                    TRUNCATE TABLE checkpoints, checkpoint_writes, checkpoint_blobs CASCADE;
                """)
                print("✅ Successfully deleted all thread checkpoints and memory!")
    except Exception as e:
        print(f"Error while clearing threads: {e}")
        # In some versions of LangGraph, the tables might be named slightly differently 
        # or `checkpoint_blobs` might not exist. Let's do a fallback:
        print("\nTrying to clear tables one by one...")
        try:
            async with AsyncConnectionPool(
                conninfo=DB_URI, max_size=1, min_size=1, kwargs={"autocommit": True}
            ) as pool:
                async with pool.connection() as conn:
                    try: await conn.execute("TRUNCATE TABLE checkpoints CASCADE;")
                    except: pass
                    try: await conn.execute("TRUNCATE TABLE checkpoint_writes CASCADE;")
                    except: pass
                    try: await conn.execute("TRUNCATE TABLE checkpoint_blobs CASCADE;")
                    except: pass
            print("✅ Successfully attempted to delete all individual thread tables!")
        except Exception as fallback_e:
            print(f"Fallback error: {fallback_e}")


if __name__ == "__main__":
    asyncio.run(clear_threads())
