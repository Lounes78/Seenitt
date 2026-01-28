import asyncio
import functools
import multiprocessing as mp
import websockets
import faulthandler

# Verifier les import gpu9 
from src.server.config import PORT, SAVE_DEBUG_FRAMES
from src.server.network import connection_handler
from plant_worker import plant_worker_process

faulthandler.enable()

async def main():
    mp.set_start_method("spawn", force=True)

    # Queues shared between network + worker
    frame_queue = mp.Queue(maxsize=3)
    result_queue = mp.Queue()

    # Start Plant Worker
    print("[PlantServer] Starting plant worker...", flush=True)
    worker = mp.Process(
        target=plant_worker_process,
        args=(frame_queue, result_queue),
        daemon=True,
    )
    worker.start()

    # Start WebSocket Server
    print(f"[PlantServer] Starting server on port {PORT}...", flush=True)

    handler = functools.partial(connection_handler, frame_queue=frame_queue)

    try:
        async with websockets.serve(
            handler,
            "0.0.0.0",
            PORT,
            ping_interval=None,
            max_size=None,
        ):
            print(f"[PlantServer] Ready on ws://0.0.0.0:{PORT}", flush=True)
            print(f"[PlantServer] SAVE_DEBUG_FRAMES={SAVE_DEBUG_FRAMES}", flush=True)

            while True:
                # Optional: read logs/messages from worker
                while not result_queue.empty():
                    print(f"[PlantWorker] {result_queue.get()}", flush=True)

                if not worker.is_alive():
                    print("[PlantServer] Worker died — shutting down.", flush=True)
                    break

                await asyncio.sleep(1.0)

    finally:
        print("[PlantServer] Shutting down...", flush=True)
        try:
            frame_queue.put(None)  # poison pill
        except Exception:
            pass

        worker.join(timeout=5)
        if worker.is_alive():
            worker.terminate()

        print("[PlantServer] Shutdown complete.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
