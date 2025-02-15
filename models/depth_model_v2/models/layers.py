from tensorflow.keras.layers import Dense, Dropout

LAYERS = {
    "Dropout1": lambda: Dropout(0.1),
    "Dropout2": lambda: Dropout(0.2),
    "Dropout3": lambda: Dropout(0.3),
    "Dropout4": lambda: Dropout(0.4),
    "Dropout5": lambda: Dropout(0.5),
    "Dropout6": lambda: Dropout(0.6),
    "D2": lambda: Dense(2, activation='relu'),
    "D4": lambda: Dense(4, activation='relu'),
    "D8": lambda: Dense(8, activation='relu'),
    "D12": lambda: Dense(12, activation='relu'),
    "D16": lambda: Dense(16, activation='relu'),
    "D24": lambda: Dense(24, activation='relu'),
    "D32": lambda: Dense(32, activation='relu'),
    "D48": lambda: Dense(48, activation='relu'),
    "D64": lambda: Dense(64, activation='relu'),
}