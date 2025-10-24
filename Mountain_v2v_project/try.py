# import tensorflow as tf
# print(tf.config.list_physical_devices('GPU'))

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
print("GPUs detected:", gpus)
print("TensorFlow will run on:", tf.test.gpu_device_name())


# replace path if different
# C:\Users\mayan\AppData\Local\Microsoft\WindowsApps\python3.exe -m venv C:\Users\mayan\Desktop\V2V\v2v_gpu_env
