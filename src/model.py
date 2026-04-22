from tensorflow.keras.layers import Input, Conv1D, Dense, GlobalAveragePooling1D
from tensorflow.keras.models import Model

def build_model(seq_len=20):
    inp = Input(shape=(seq_len,1))

    x = Conv1D(32,3,activation='relu',padding='causal')(inp)
    x = Conv1D(32,3,activation='relu',padding='causal')(x)
    x = Conv1D(64,3,activation='relu',padding='causal')(x)

    x = GlobalAveragePooling1D()(x)

    x = Dense(64,activation='relu')(x)
    x = Dense(32,activation='relu')(x)

    out = Dense(1,activation='sigmoid')(x)

    model = Model(inp, out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model