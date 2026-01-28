import asyncio
import websockets
import functools
import multiprocessing as mp
import faulthandler

from src.server.config import PORT, SAVE_DEBUG_FRAMES
from src.server.worker import ai_worker_process
from src.server.network import connection_handler

faulthandler.enable()

async def main():
    mp.set_start_method('spawn', force=True)
    
    # 1. Setup Queues
    frame_queue = mp.Queue(maxsize=1) 
    result_queue = mp.Queue()
    
    # Queue for sending data BACK to the websocket client
    # Using Manager Queue to ensure safe IPC if needed, though mp.Queue is usually fine here
    manager = mp.Manager()
    response_queue = manager.Queue()
    
    # 2. Start AI Worker
    print("[Main] Starting AI worker...", flush=True)
    p = mp.Process(target=ai_worker_process, args=(frame_queue, result_queue))
    p.start()
    
    # 3. Start Server
    print(f"[Main] Server starting on port {PORT}...", flush=True)
    try:
        # Bind the queues to the handler
        handler = functools.partial(connection_handler, frame_queue=frame_queue, response_queue=response_queue)
        
        async with websockets.serve(handler, "0.0.0.0", PORT, ping_interval=None, max_size=None):
            print(f"[Main] Ready on ws://0.0.0.0:{PORT}")
            
            while True:
                # ROUTING LOOP: Worker -> Websocket
                while not result_queue.empty(): 
                    msg = result_queue.get()
                    
                    if isinstance(msg, dict) and msg.get('type') == 'board_state':
                        # Route FEN to the network handler
                        response_queue.put(msg)
                    else:
                        # Log standard messages
                        print(f"[Main] {msg}", flush=True)
                
                # Health Check
                if not p.is_alive():
                    print("[Main] Worker died! Exiting...", flush=True)
                    break
                    
                await asyncio.sleep(0.1)
                
    finally:
        print("[Main] Shutting down...", flush=True)
        try: frame_queue.put(None)
        except: pass
        p.join(timeout=5)
        if p.is_alive(): p.terminate()
        print("[Main] Shutdown complete.")

if __name__ == "__main__":
    asyncio.run(main())



# """
# Streaming Chessboard Detection Server
# Entry Point
# """
# import asyncio
# import websockets
# import functools
# import multiprocessing as mp
# import faulthandler

# from src.server.config import PORT, SAVE_DEBUG_FRAMES
# from src.server.worker import ai_worker_process
# from src.server.network import connection_handler

# faulthandler.enable()

# async def main():
#     mp.set_start_method('spawn', force=True)
    
#     # 1. Setup Queues
#     frame_queue = mp.Queue(maxsize=3) 
#     result_queue = mp.Queue()
    
#     # 2. Start AI Worker
#     print("[Main] Starting AI worker...", flush=True)
#     p = mp.Process(target=ai_worker_process, args=(frame_queue, result_queue))
#     p.start()
    
#     # 3. Start Server
#     print(f"[Main] Server starting on port {PORT}...", flush=True)
#     try:
#         handler = functools.partial(connection_handler, frame_queue=frame_queue)
#         async with websockets.serve(handler, "0.0.0.0", PORT, ping_interval=None, max_size=None):
#             print(f"[Main] Ready on ws://0.0.0.0:{PORT}")
#             print(f"[Main] Frame Saving: {SAVE_DEBUG_FRAMES}")
            
#             while True:
#                 # Log messages from worker
#                 while not result_queue.empty(): 
#                     print(f"[Main] {result_queue.get()}", flush=True)
                
#                 # Health Check
#                 if not p.is_alive():
#                     print("[Main] Worker died! Exiting...", flush=True)
#                     break
                    
#                 await asyncio.sleep(1.0)
                
#     finally:
#         print("[Main] Shutting down...", flush=True)
#         try:
#             frame_queue.put(None) # Poison pill
#         except:
#             pass
#         p.join(timeout=5)
#         if p.is_alive(): p.terminate()
#         print("[Main] Shutdown complete.")

# if __name__ == "__main__":
#     asyncio.run(main())
