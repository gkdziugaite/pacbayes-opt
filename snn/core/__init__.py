# Retrieve the package location
import os
import snn
import inspect
package_path = os.path.dirname(inspect.getfile(snn))

import tensorflow as tf
config = tf.ConfigProto()

