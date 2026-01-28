import asyncio
import websockets
import av
import cv2
import os
import uuid
import json
import traceback
import queue
from src.server.config import SAVE_DEBUG_FRAMES, SAVE_DIR

# Shared Lock to prevent multiple clients
ACTIVE_LOCK = asyncio.Lock()

def decode_safe(codec, message):
    try:
        packets = codec.parse(message)
        decoded_frames = []
        for packet in packets:
            frames = codec.decode(packet)
            for frame in frames:
                decoded_frames.append(frame.to_ndarray(format='bgr24'))
        return decoded_frames
    except Exception as e:
        return e

async def watchdog_task(conn_id, watchdog_state):
    last_count = 0
    stall_cycles = 0
    while watchdog_state['running']:
        await asyncio.sleep(2.0)
        current_count = watchdog_state['message_count']
        if current_count == last_count:
            stall_cycles += 1
            if stall_cycles >= 2:
                print(f"[Watchdog] #{conn_id} Stalled for {stall_cycles * 2}s.", flush=True)
        else:
            if stall_cycles >= 2:
                print(f"[Watchdog] #{conn_id} Resumed.", flush=True)
            stall_cycles = 0
            last_count = current_count

async def connection_handler(websocket, frame_queue, response_queue):
    # Depending on websockets version, 'path' might be passed as 2nd arg. 
    # We ignore *args to be safe across versions.
    conn_id = str(uuid.uuid4())[:4]
    print(f"[Net] #{conn_id} Connected from {websocket.remote_address}.", flush=True)
    
    if SAVE_DEBUG_FRAMES:
        os.makedirs(SAVE_DIR, exist_ok=True)

    if ACTIVE_LOCK.locked():
        print(f"[Net] #{conn_id} Rejected: Server busy.", flush=True)
        await websocket.close(1013, "Busy")
        return

    async with ACTIVE_LOCK:
        codec = av.CodecContext.create('h264', 'r')
        loop = asyncio.get_running_loop()
        watchdog_state = {'running': True, 'message_count': 0}
        watchdog = asyncio.create_task(watchdog_task(conn_id, watchdog_state))
        
        # State containers for the loops
        state = {'local_frame_count': 0}

        # --- TASK 1: RECEIVE VIDEO ---
        async def receive_loop():
            try:
                async for message in websocket:
                    watchdog_state['message_count'] += 1
                    result = await loop.run_in_executor(None, decode_safe, codec, message)
                    
                    if isinstance(result, Exception) or not result:
                        continue

                    if frame_queue.qsize() > 2:
                        print(f"[Net] #{conn_id} Drop (Queue Full)", flush=True)
                        state['local_frame_count'] += len(result)
                        continue

                    for img in result:
                        state['local_frame_count'] += 1
                        
                        if SAVE_DEBUG_FRAMES:
                            fname = os.path.join(SAVE_DIR, f"frame_{state['local_frame_count']:05d}.jpg")
                            await loop.run_in_executor(None, cv2.imwrite, fname, img)

                        try:
                            frame_queue.put_nowait((state['local_frame_count'], img))
                        except: pass
            except websockets.exceptions.ConnectionClosed:
                print(f"[Net] #{conn_id} RX Closed.")
            except Exception as e:
                print(f"[Net] #{conn_id} RX Error: {e}")

        # --- TASK 2: SEND FEN ---
        async def send_loop():
            # FIXED: Removed 'while not websocket.closed' which causes AttributeError
            while True: 
                try:
                    # Non-blocking check for new data
                    if not response_queue.empty():
                        data = response_queue.get_nowait()
                        await websocket.send(json.dumps(data))
                    else:
                        await asyncio.sleep(0.05) # Yield to event loop
                except queue.Empty:
                    await asyncio.sleep(0.05)
                except websockets.exceptions.ConnectionClosed:
                    print(f"[Net] #{conn_id} TX Closed.")
                    break
                except Exception as e:
                    print(f"[Net] #{conn_id} TX Error: {e}")
                    break

        # --- RUN BOTH ---
        # We use asyncio.wait with FIRST_COMPLETED.
        # If the client disconnects, receive_loop finishes, and we immediately cancel send_loop.
        try:
            rx_task = asyncio.create_task(receive_loop())
            tx_task = asyncio.create_task(send_loop())

            done, pending = await asyncio.wait(
                [rx_task, tx_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            for task in pending:
                task.cancel()

        except Exception as e:
            print(f"[Net] #{conn_id} Main Error: {e}")
            traceback.print_exc()
        finally:
            watchdog_state['running'] = False
            watchdog.cancel()
            print(f"[Net] #{conn_id} Disconnected.")


# # src/server/network.py

# import asyncio
# import websockets
# import av
# import cv2
# import os
# import uuid
# import traceback
# from src.server.config import SAVE_DEBUG_FRAMES, SAVE_DIR

# # Shared Lock to prevent multiple clients
# ACTIVE_LOCK = asyncio.Lock()

# def decode_safe(codec, message):
#     try:
#         packets = codec.parse(message)
#         decoded_frames = []
#         for packet in packets:
#             frames = codec.decode(packet)
#             for frame in frames:
#                 decoded_frames.append(frame.to_ndarray(format='bgr24'))
#         return decoded_frames
#     except Exception as e:
#         return e

# async def watchdog_task(conn_id, watchdog_state):
#     last_count = 0
#     stall_cycles = 0
#     while watchdog_state['running']:
#         await asyncio.sleep(2.0)
#         current_count = watchdog_state['message_count']
#         if current_count == last_count:
#             stall_cycles += 1
#             if stall_cycles >= 2:
#                 print(f"[Watchdog] #{conn_id} Stalled for {stall_cycles * 2}s.", flush=True)
#         else:
#             if stall_cycles >= 2:
#                 print(f"[Watchdog] #{conn_id} Resumed.", flush=True)
#             stall_cycles = 0
#             last_count = current_count

# async def connection_handler(websocket, frame_queue):
#     conn_id = str(uuid.uuid4())[:4]
#     print(f"[Net] #{conn_id} Connected from {websocket.remote_address}.", flush=True)
    
#     if SAVE_DEBUG_FRAMES:
#         os.makedirs(SAVE_DIR, exist_ok=True)

#     if ACTIVE_LOCK.locked():
#         print(f"[Net] #{conn_id} Rejected: Server busy.", flush=True)
#         await websocket.close(1013, "Busy")
#         return

#     async with ACTIVE_LOCK:
#         codec = av.CodecContext.create('h264', 'r')
#         loop = asyncio.get_running_loop()
#         local_frame_count = 0 
#         message_count = 0
        
#         watchdog_state = {'running': True, 'message_count': 0}
#         watchdog = asyncio.create_task(watchdog_task(conn_id, watchdog_state))
        
#         try:
#             async for message in websocket:
#                 message_count += 1
#                 watchdog_state['message_count'] = message_count
                
#                 result = await loop.run_in_executor(None, decode_safe, codec, message)
                
#                 if isinstance(result, Exception):
#                     continue
#                 if not result:
#                     continue

#                 if frame_queue.qsize() > 2:
#                     print(f"[Net] #{conn_id} Drop (Queue Full)", flush=True)
#                     local_frame_count += len(result)
#                     continue

#                 for img in result:
#                     local_frame_count += 1
                    
#                     if SAVE_DEBUG_FRAMES:
#                         fname = os.path.join(SAVE_DIR, f"frame_{local_frame_count:05d}.jpg")
#                         await loop.run_in_executor(None, cv2.imwrite, fname, img)

#                     try:
#                         frame_queue.put_nowait((local_frame_count, img))
#                         print(f"[Net] Queued Frame {local_frame_count}", flush=True)
#                     except:
#                         pass
            
#             print(f"[Net] #{conn_id} Disconnected.", flush=True)

#         except websockets.exceptions.ConnectionClosed:
#             print(f"[Net] #{conn_id} Closed.", flush=True)
#         except Exception as e:
#             print(f"[Net] #{conn_id} Error: {e}", flush=True)
#             traceback.print_exc()
#         finally:
#             watchdog_state['running'] = False
#             watchdog.cancel()
