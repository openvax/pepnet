try:
    import tensorflow as tf
    # TensorFlow flooding the screen with info log statements
    tf.logging.set_verbosity(tf.logging.ERROR)
except ImportError:
    pass

__version__ = "0.0.1"
