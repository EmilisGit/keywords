import asyncio
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

SAMPLE_RATE = 16000
CHUNK_DURATION = 1.0
OVERLAP = 0.5
BUFFER_SIZE = int(SAMPLE_RATE * CHUNK_DURATION) 
STEP_SIZE = int(SAMPLE_RATE * (CHUNK_DURATION - OVERLAP))
MODEL_PATH = "sound_instructions.keras"
audio_model = tf.keras.models.load_model(MODEL_PATH)

def get_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

label_names= ['down', 'go', 'left', 'no', 'right', 'silence', 'stop', 'up', 'yes']

class ExportModel(tf.Module):
  def __init__(self, model):
    self.model = model

    # Accept either a string-filename or a batch of waveforms.
    self.__call__.get_concrete_function(
        x=tf.TensorSpec(shape=(), dtype=tf.string))
    self.__call__.get_concrete_function(
       x=tf.TensorSpec(shape=[None, 16000], dtype=tf.float32))


  @tf.function
  def __call__(self, x):
    # If a string is passed, load the file and decode it. 
    if x.dtype == tf.string:
      x = tf.io.read_file(x)
      x, _ = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000,)
      x = tf.squeeze(x, axis=-1)
      x = x[tf.newaxis, :]
    
    x = get_spectrogram(x)  
    result = self.model(x, training=False)
    
    class_ids = tf.argmax(result, axis=-1)
    class_names = tf.gather(label_names, class_ids)
    return {'predictions':result,
            'class_ids': class_ids,
            'class_names': class_names}

instructions_model = ExportModel(audio_model)

app = FastAPI()
model_path = "sound_instructions.keras"
audio_model = tf.keras.models.load_model(model_path)

@app.websocket("/ws/audio")
async def audio_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    audio_buffer = bytearray()
    
    try:
        while True:
            data = await websocket.receive_bytes()
            audio_buffer.extend(data)
            
            while len(audio_buffer) >= BUFFER_SIZE * 2:
                
                window_bytes = audio_buffer[:BUFFER_SIZE * 2]
                audio_np = np.frombuffer(window_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                model_input = audio_np.reshape(1, 16000)
                result = instructions_model(model_input)
                
                class_name = result['class_names'].numpy()[0].decode('utf-8')
                await websocket.send_json({"detected": "class_name"})

                del audio_buffer[:STEP_SIZE * 2]
                
    except Exception as e:
        print(f"Connection closed: {e}")