import sys
print(f"Python version: {sys.version}")
try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
    import tf_keras as keras
    print("TF-Keras loaded successfully")
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    print("All imports successful")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
