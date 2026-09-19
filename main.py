import numpy as np
import pandas as pd
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from WhatevPlot import WhatevPlot as wp
from NNFS import NNFS
import sys

def get_random_batch_from_csv(df, batch_size=128, num_classes=10):
    batch_df = df.sample(n=batch_size)
    labels = batch_df["label"].to_numpy()
    inputs = batch_df.drop(columns=["label"]).to_numpy() / 255.0
    y = np.eye(num_classes)[labels]
    return inputs, y

if __name__ == "__main__":

    csv_file_path = "Data/train.csv"

    df = pd.read_csv(csv_file_path)
    df = df.sample(frac=1, random_state=32).reset_index(drop=True)
    split = int(0.9 * len(df))

    train_df = df.iloc[:split]
    test_df  = df.iloc[split:]

    nn = NNFS([784, 64, 64, 10], 0.1)

    inputs_test = test_df.drop(columns=["label"]).to_numpy() / 255.0
    labels_test = test_df["label"].to_numpy()
    y_test = np.eye(10)[labels_test]

    app = QApplication(sys.argv)
    w1 = wp(bufferSize=200, title="loss")
    w2 = wp(bufferSize=200, title="accuracy")
    alp = 0.99

    for iteration in range(1000):
        inputs, y = get_random_batch_from_csv(train_df, batch_size=256, num_classes=10)
        
        nn.forward(inputs)
        loss, acc = nn.loss(y)

        nn.backward()
        nn.optimize()

        nn.forward(inputs_test)
        loss, acc = nn.loss(y_test)
        print(f"{loss:.4f}", f"{acc:.4f}", f"{nn.optimizer.currentAlpha:.4f}", "iteration:", iteration)
        w1.display(w1.data[-1] * alp + loss * (1-alp))
        w2.display(w2.data[-1] * alp + acc * (1-alp))
        QApplication.processEvents()

    sys.exit(app.exec())