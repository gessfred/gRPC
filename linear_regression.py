import numpy as  np
import torch
numberOfSamples = 1000
numberOfFeatures = 1
x = np.random.random(numberOfSamples).reshape(-1, 1)
#y = (np.random.random(numberOfSamples) > 0.5).astype(int)
x = torch.from_numpy(x).float()
y = x * 3 + 5 + 2 * torch.rand(numberOfSamples)

#targets = torch.from_numpy(targets)

# Define the model
def model(x, w, b):
    return x @ w.t() + b

# MSE loss
def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()

def batch_iter(y, tx, batch_size, num_batches=1):
    data_size = len(y)
    shuffle_indices = np.random.permutation(np.arange(data_size))
    shuffled_y = y[shuffle_indices]
    shuffled_tx = tx[shuffle_indices]
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def sgd(targets, inputs, batch_size, max_iter, λ=1e-2):
    losses = []
    w = torch.randn(1, numberOfFeatures, requires_grad=True)
    b = torch.randn(numberOfFeatures, requires_grad=True)
    acc_loss = 0
    i = 0
    for ybatch, xbatch in batch_iter(targets, inputs, batch_size, max_iter):
        preds = model(xbatch, w, b)
        loss = mse(preds, ybatch)
        print('epoch', i, " loss=", loss)
        loss.backward()
        with torch.no_grad():
            w -= w.grad * λ
            b -= b.grad * λ
            w.grad.zero_()
            b.grad.zero_()
        i += 1
    return w, b
"""def train(inputs, targets, λ=1e-5):
    [l, h] = inputs.shape
    print(inputs.shape)
    # Weights and biases
    w = torch.randn(2, h, requires_grad=True)
    b = torch.randn(2, requires_grad=True)
    # Train for 100 epochs
    for i in range(100):
        preds = model(inputs, w, b)
        loss = mse(preds, targets)
        print('epoch', i, ' loss=', loss, 'accuracy=', np.sum((preds == targets).astype(int)))
        loss.backward()
        with torch.no_grad():
            w -= w.grad * λ
            b -= b.grad * λ
            w.grad.zero_()
            b.grad.zero_()
    return w, b"""

print(sgd(y, x, 10, 100))
