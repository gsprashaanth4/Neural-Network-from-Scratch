import numpy as np
import pandas as pd

from NN import NN
from NN import Optimizer_SGD
from NN import Optimizer_ADAGRAD
from NN import Optimizer_RMSprop
from NN import Optimizer_ADAM

import matplotlib.pyplot as plt
import time

def get_random_batch_from_csv(df, batch_size=64, num_classes=10):
    
    # 2. Randomly sample 64 rows from the dataframe
    batch_df = df.sample(n=batch_size)
    
    # 3. Separate features (pixels) and labels
    # Extract labels and convert to a 1D NumPy array
    labels = batch_df["label"].to_numpy()
    
    # Extract pixel values, drop the label column, and convert to NumPy array
    # We divide by 255.0 to normalize pixel values between 0 and 1
    inputs = batch_df.drop(columns=["label"]).to_numpy() / 255.0
    
    # 4. Convert 1D labels array into a One-Hot Encoded Matrix
    # np.eye(10) creates an identity matrix; indexing it creates the one-hot rows
    y = np.eye(num_classes)[labels]
    
    return inputs.T, y.T

if __name__ == "__main__":

    csv_file_path = "Data/train.csv"

    df = pd.read_csv(csv_file_path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split = int(0.9 * len(df))

    train_df = df.iloc[:split]
    test_df  = df.iloc[split:]

    nn = NN(in_d  = 784,
            d1    = 64,
            d2    = 64,
            d3    = 64,
            o_d   = 10)
    
    nn.optimizer = Optimizer_ADAM(
        alpha    = 0.001,
        decay    = 0,
        epsilon  = 1e-7,
        momentum = 0.99,
        rho      = 0.99
    )

    inputs_test = test_df.drop(columns=["label"]).to_numpy().T / 255.0
    labels_test = test_df["label"].to_numpy()
    y_test = np.eye(10)[labels_test].T

    loss = np.array([])
    accu = np.array([])

    loss_u = np.array([])
    accu_u = np.array([])
    time_delta_x = np.array([])

    plt.ion() # Turn on interactive mode
    fig, ax = plt.subplots()

    ax.plot(np.array([2.5,]), np.array([0,]))
    plt.pause(0.001)
    time.sleep(10)

    time_start = time.perf_counter()
    alpha = 0.9

    for epoch in range(1000):
        inputs, y = get_random_batch_from_csv(train_df, batch_size=64, num_classes=10)
        loss_val = nn.forward(inputs, y)

        preds = np.argmax(nn.activation_loss_l.softed, axis=0)
        truth = np.argmax(y, axis=0)
        acc = np.mean(preds == truth)

        nn.backward()
        nn.update()

        # if epoch % 500 == 0:
        print(f"{loss_val:.4f}", f"{acc:.4f}", f"{nn.optimizer.current_alpha:.4f}", "epoch:", epoch)

        time_stamp = time.perf_counter() - time_start

        if (len(loss) == 0):
            loss = np.append(loss, loss_val)
            accu = np.append(accu, acc)
        else:
            loss = np.append(loss, (loss[-1]) * alpha  +  loss_val * (1-alpha) )
            accu = np.append(accu, (accu[-1]) * alpha  +      acc  * (1-alpha) )

        loss_u = np.append(loss_u, loss_val)
        accu_u = np.append(accu_u, acc)
        
        time_delta_x = np.append(time_delta_x, time_stamp)

        loss_mark = np.full(len(time_delta_x), loss[-1])
        accu_mark = np.full(len(time_delta_x), accu[-1])

        ax.clear() # Clear the old lines from the window
        ax.plot(time_delta_x, loss_u, label='Loss_unfiltered', color='orange')
        ax.plot(time_delta_x, loss, label='Loss', color='red', linewidth=3)

        ax.plot(time_delta_x, accu_u, label='Accuracy_unfiltered', color='lightgreen')
        ax.plot(time_delta_x, accu, label='Accuracy', color='green', linewidth=3)

        ax.plot(time_delta_x, loss_mark, color='red', linestyle='--')
        ax.plot(time_delta_x, accu_mark, color='green', linestyle='--')

        ax.legend(loc='upper left')

        plt.pause(0.001)

loss = nn.forward(inputs_test, y_test)
preds = np.argmax(nn.activation_loss_l.softed, axis=0)
acc = np.mean(preds == labels_test)
print(f"Test Accuracy: {acc:.4f}")

plt.ioff() # Turn off interactive mode when done
plt.show() # Keep the final plot open at the end