
from keras import layers
from select_feature import data_x,data_y

from tensorflow import keras

from keras.optimizers import Adam

from keras.layers import *

import tensorflow as tf

# Read Training Data
from sklearn.model_selection import train_test_split
TEST_SIZE = 0.15
RANDOM_STATE = 0
X_train, X_test, y_train, y_test = train_test_split(data_x, data_y,
                                                    test_size=TEST_SIZE,
                                                    random_state=RANDOM_STATE)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  test_size=TEST_SIZE,
                                                  random_state=RANDOM_STATE)

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.5):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out = self.layernorm2(out1 + ffn_output)
        return out


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = tf.reshape(x, [-1, maxlen, embed_dim])
        out = x + positions
        return out

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau

def create_callbacks(best_model_filepath, tensorboard_logs_filepath):
    callback_checkpoint = ModelCheckpoint(filepath=best_model_filepath,
                                          monitor='val_loss',
                                          verbose=0,
                                          save_weights_only=False,
                                          save_best_only=False)

    callback_early_stopping = EarlyStopping(monitor='val_loss',
                                            patience=80,
                                            verbose=1)

    callback_tensorboard = TensorBoard(log_dir=tensorboard_logs_filepath,
                                       histogram_freq=0,
                                       write_graph=False)

    callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                           factor=0.1,
                                           min_lr=1e-4,
                                           patience=0,
                                           verbose=1)

    return [callback_checkpoint, callback_early_stopping,
            callback_tensorboard, callback_reduce_lr]
num_heads = 5   # Number of attention heads
ff_dim = 64     # Hidden layer size in feed forward network inside transformer
embed_dim = 36
maxlen = 1
inputs = Input(embed_dim*maxlen)
embedding_layer = TokenAndPositionEmbedding(maxlen, embed_dim)

# Encoder Architecture
x = embedding_layer(inputs)
transformer_block_1 = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)
transformer_block_2 = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)
x = transformer_block_1(x)
x = transformer_block_2(x)
flatten_layer = layers.Flatten()
# Output
x = flatten_layer(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
best_model_filepath = "Transformers_Model.ckpt"
tensorboard_logs_filepath = "./Trans_logs/"
model.compile(optimizer=Adam(0.001),
              loss="binary_crossentropy",
              metrics=['accuracy', 'Recall', 'Precision'])
model.summary()
history_transformer = model.fit(
    X_train,y_train, batch_size=64, epochs=300, validation_data=(X_test,y_test),callbacks= create_callbacks(best_model_filepath,tensorboard_logs_filepath))
import matplotlib.pyplot as plt

def plot_progress(history_dict):
    for key in list(history_dict.keys())[:4]:
        plt.clf()  # Clears the figure
        training_values = history_dict[key]
        val_values = history_dict['val_' + key]
        epochs = range(1, len(training_values) + 1)

        plt.plot(epochs, training_values, 'bo', label='Training ' + key)

        plt.plot(epochs, val_values, 'b', label='Validation ' + key)

        if key != 'loss':
            plt.ylim([0., 1.1])

        plt.title('Training and Validation ' + key)
        plt.xlabel('Epochs')
        plt.ylabel(key)
        plt.legend()
        plt.show()

plot_progress(history_transformer.history)