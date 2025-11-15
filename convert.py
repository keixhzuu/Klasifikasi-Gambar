import sys
import types
import warnings
warnings.filterwarnings("ignore")

#  PATCH buat JAX & TFDF biar gak error 
class FakePolyShape:
    def __init__(self, *args, **kwargs):
        pass

fake_shape_poly = types.ModuleType("jax.experimental.jax2tf.shape_poly")
fake_shape_poly.PolyShape = FakePolyShape

sys.modules["tensorflow_decision_forests"] = types.ModuleType("tensorflow_decision_forests")
sys.modules["jax"] = types.ModuleType("jax")
sys.modules["jax.experimental"] = types.ModuleType("jax.experimental")
sys.modules["jax.experimental.jax2tf"] = types.ModuleType("jax.experimental.jax2tf")
sys.modules["jax.experimental.jax2tf.shape_poly"] = fake_shape_poly

# --- Import utama ---
import tensorflow as tf
import tensorflowjs as tfjs

model_path = "best_model.keras"   
output_path = "tfjs_model"       

print("📂 Step 1: Load model Keras dari:", model_path)
model = tf.keras.models.load_model(model_path)
print("✅ Model berhasil dimuat!")

print("⚙️ Step 2: Konversi ke format TensorFlow.js...")
tfjs.converters.save_keras_model(model, output_path)
print(f"✅ Konversi selesai! Model tersimpan di: {output_path}")
