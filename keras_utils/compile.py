from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, Activation, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.initializers import glorot_uniform
from tensorflow import random as tf_random
import random
import numpy as np

# custom functions
class MultiplyScalar(Lambda):
    def __init__(self, scalar, **kwargs):
        super().__init__(lambda x: x * scalar, **kwargs)

class AddScalar(Lambda):
    def __init__(self, scalar, **kwargs):
        super().__init__(lambda x: x + scalar, **kwargs)


seed_value = 12345
np.random.seed(seed_value)
tf_random.set_seed(seed_value)
random.seed(seed_value)
initializer = glorot_uniform(seed=seed_value)

inputs = [
  #$INPUTS_PLACEHOLDER
]

all_layers = [
  #$LAYERS_PLACEHOLDER
]

#$GRAPH_PLACEHOLDER

outputs = [
  #$OUTPUTS_PLACEHOLDER
]

optimizer="adam"

#$OPTIMIZER_PLACEHOLDER

# Create the model
model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=optimizer, loss='mse')

model.summary()

#$MODEL_NAME_PLACEHOLDER = model