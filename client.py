import asyncio
import sounddevice as sd
import numpy as np
import websockets
import argparse
import sys
import json

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCKSIZE = 1024

async def audio_sender(uri):
    audio_queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def callback(indata, frames, time, status):
        if status:
            print(f"[WARN] {status}", file=sys.stderr)
        
        data_int16 = (indata[:, 0] * 32767).astype(np.int16)
        
        loop.call_soon_threadsafe(audio_queue.put_nowait, data_int16.tobytes())

    print(f"[INFO] Connecting to {uri}...")
    
    async with websockets.connect(uri) as websocket:
        print("[INFO] Connected.")

        async def receive_handler():
            try:
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        detected = data.get("detected", "unknown")
                        confidence = data.get("confidence", None)
                        inference_time = data.get("inference_time", None)
                        print(f"[SERVER] Detected: {detected}, Confidence: {confidence}, Inference Time: {inference_time}")
                    except json.JSONDecodeError:
                        print(f"[SERVER] Raw: {message}")
            except websockets.exceptions.ConnectionClosed:
                print("[INFO] Server closed connection.")
            except Exception as e:
                print(f"[ERROR] Receiver error: {e}")

        receive_task = asyncio.create_task(receive_handler())

        print("[INFO] Starting microphone stream...")
        with sd.InputStream(samplerate=SAMPLE_RATE,
                            channels=CHANNELS,
                            blocksize=BLOCKSIZE,
                            dtype='float32',
                            callback=callback):
            
            print("[INFO] Streaming. Press Ctrl+C to stop.")
            
            try:
                while True:
                    audio_data = await audio_queue.get()
                    await websocket.send(audio_data)
            finally:
                receive_task.cancel()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("uri", type=str, help="WebSocket URI")
    args = parser.parse_args()

    try:
        asyncio.run(audio_sender(args.uri))
    except KeyboardInterrupt:
        print("\n[INFO] Exiting...")