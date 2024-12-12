from tensorflow.keras.layers import Dense

LAYERS = {
    "D4": lambda: Dense(4, activation='relu'),
    "D8": lambda: Dense(8, activation='relu'),
    "D16": lambda: Dense(16, activation='relu'),
    "D24": lambda: Dense(24, activation='relu'),
    "D32": lambda: Dense(32, activation='relu'),
    "D64": lambda: Dense(64, activation='relu'),
}