from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
import numpy as np

fake_input = Input(1,)
all_layers = []
inputs = []
outputs = []
model = Model(inputs=fake_input, outputs=Dense(2)(fake_input))

#$SCRIPT_START
from keras.callbacks import EarlyStopping

def preprocess_dynamic_data(data_stream, data_lengths):
    # Calculate the total length of the data segments
    total_length = sum(data_lengths)
    
    # Calculate the number of samples in the data stream
    num_samples = len(data_stream) // total_length
    
    # Reshape the stream into a 2D array where each row is a sample
    samples = np.reshape(data_stream, (num_samples, total_length))
    
    # Initialize cursor to keep track of where we are in the sample
    cursor = 0
    separated_data = []
    for length in data_lengths:
        # Slice out each data segment
        separated_data.append(samples[:, cursor:cursor + length])
        cursor += length
    
    return separated_data

#$MODEL_NAME_PLACEHOLDER

weights = None
#read binary fine with float32 vector
with open('weights/temp_weights.bin', 'rb') as f:
  weights = f.read()


weights = np.frombuffer(weights, dtype=np.float32)

data_x = None
with open('data_buffer/temp_data_x.bin', 'rb') as f:
  data_x = f.read()

data_x = preprocess_dynamic_data(np.frombuffer(data_x, dtype=np.float32), [input_layer.shape[1] for input_layer in model.inputs])

# print(data_x)


data_y = None
with open('data_buffer/temp_data_y.bin', 'rb') as f:
  data_y = f.read()

data_y = preprocess_dynamic_data(np.frombuffer(data_y, dtype=np.float32), [output_layer.shape[1] for output_layer in model.outputs])

# print(data_y)



if np.any(weights != 0):
    for layer in model.layers:
        if hasattr(layer, 'weights') and len(layer.weights) > 0:
            weights_size = layer.weights[0].shape[0] * layer.weights[0].shape[1]
            layer_weights = weights[:weights_size]
            layer_weights = layer_weights.reshape(layer.weights[0].shape)
            layer.set_weights([layer_weights, np.zeros(layer.weights[1].shape)])
            weights = weights[weights_size:]
        else:
            continue
        if hasattr(layer, 'bias'):
            layer_bias = weights[:layer.bias.shape[0]]
            layer.set_weights([layer.get_weights()[0], layer_bias])
            weights = weights[layer.bias.shape[0]:]
        
    assert(len(weights) == 0)

epochs=20 
batch_size=1 

#$EPOCHS_PLACEHOLDER
#$BATCH_SIZE_PLACEHOLDER

early_stopping_callback = EarlyStopping(
    monitor='loss',  # monitor training loss
    patience=epochs,     # number of epochs with no improvement after which training will be stopped
    mode='min',      # since we want to minimize loss
    verbose=1,
    restore_best_weights=True  # restores model weights from the epoch with the best value of the monitored quantity
)



model.fit(
    data_x, 
    data_y, 
    epochs=epochs, 
    batch_size=batch_size, 
    verbose=0,
    callbacks=early_stopping_callback
)



# Save weights to a binary file
weights = []
for layer in model.layers:
    if hasattr(layer, 'weights') and len(layer.weights) > 0:
        layer_weights = layer.get_weights()[0].flatten()
        weights.extend(layer_weights)
    else:
        continue
    if hasattr(layer, 'bias'):
        layer_bias = layer.get_weights()[1].flatten()
        weights.extend(layer_bias)

weights = np.array(weights, dtype=np.float32)
with open('weights/temp_weights.bin', 'wb') as f:
    f.write(weights.tobytes())
      