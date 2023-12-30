from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


import numpy as np


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