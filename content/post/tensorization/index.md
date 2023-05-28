---
title: Tensorization of neural network layers
# subtitle: Welcome 👋 We know that first impressions are important, so we've populated your new site with some initial content to help you get familiar with everything in no time.

# Summary for listings and search engines
summary: Implementing tensorization with einstein summation notation, as part of the invited talk at CYI, 2023. 

# Link this post with a project
projects: []

# Date published
date: "2023-05-23T00:00:00Z"

# Date updated
date: "2023-05-23T00:00:00Z"

# Is this an unpublished draft?
draft: false

# Show this page in the Featured widget?
featured: false

# Featured image
# Place an image named `featured.jpg/png` in this page's folder and customize its options here.
image:
  caption: 'Preview'
  focal_point: ""
  placement: 2
  preview_only: false

authors:
- admin

tags:
- Academic, Tensorization

categories:
- Talk
---

## Introduction

First proposed by Novikov et al. 2015, tensorization of neural network aims to represent large weight matrices in neural networks with product of smaller, high dimensional tensors, which largely reduces the computational cost. For more details and applications please refer to my [talk](../../../static/uploads/CYI_Talk_2023.pdf). In this post, we see how easy it has become to implement such tensorized neural networks. 

## 1. Einstein summation notation. 

Einstein summation notation can express tensor contraction in an extremely elegant fashion. There are only two conventions to follow: 
- The mode ("dimensionality") of a tensor is denoted using indices, which is in contrast to the usual matrix notation, where one denotes an entry in a tensor with the indices.
- The contraction is carried out over all modes whose indices appear more than once. Note: the order in which the contraction of multiple modes take place is not denoted explicitly. 

One could express a large amount of tensor operations using Einstein summation notation in numpy, torch, tensorflow and other libraries.

Here are some basic usage: 

```python
import numpy as np
import torch

np.random.seed(123)
torch.manual_seed(123)
```

#### 1.1. Summation over all elements
```python
a = torch.randn(5)

torch.einsum('n->', a)

# equivalent to a.sum()

```

#### 1.2. Summation over rows or columns
```python
A = torch.randn(5, 3)

torch.einsum('nm->m', A)

# equivalent to A.sum(axis=0)

```

#### 1.3. Inner product of vectors
```python
a = torch.randn(5)
b = torch.randn(5)

torch.einsum('n,n->', a, b)

# equivalent to torch.inner(a, b)

```

#### 1.4. Outer product of vectors
```python
a = torch.randn(5)
b = torch.randn(3)

torch.einsum('n,m->nm', a, b) 

# equivalent to torch.outer(a, b)

```

#### 1.5. Matrix vector product
```python
A = torch.randn(3, 5)
b = torch.randn(5)

torch.einsum('mn,n->m', A, b) 

# equivalent to torch.matmul(A, b)

```

#### 1.6. Matrix matrix product
```python
A = torch.randn(3, 5)
B = torch.randn(5, 4)

torch.einsum('mn,nk->mk', A, B) 

# equivalent to torch.matmul(A, B)

```

#### 1.7. Tensor contraction
```python

A = torch.randn(5, 10, 7)
B = torch.randn(5, 7, 2)

torch.einsum('mnk,mkr->nr', A, B)

```

which is equivalent to
```python
C = torch.zeros(10, 2)

for m in range(5):
    for n in range(10):
        for k in range(7):
            for r in range(2):
                C[n, r] += A[m, n, k] * B[m, k, r]

```

or 

```python
C = torch.zeros(10, 2)

for n in range(10):
    for r in range(2):
        C[n, r] = (A[:, n, :] * B[:, :, r]).sum()

```

## 2. Tensorizing neural network layers

We create a simple regression task as an isolated layer. 

```python
import numpy as np
import torch
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(123)
torch.manual_seed(123)
```

### 2.1. Differentiability

Obviously, tensor contraction are differentiable linear operations. The `einsum` API is thus also differentiable. 

```python
W = torch.randn(3, 5, requires_grad=True)
x = torch.randn(5)
y = torch.randn(3)

y_hat = torch.einsum('NM,M->N', W, x)
loss = ((y - y_hat)**2).sum()
loss.backward()

W.grad

# tensor([[ 2.3307,  2.4076,  4.2191, -3.5718, -4.3023],
#         [ 3.2671,  3.3749,  5.9143, -5.0069, -6.0309],
#         [-4.2287, -4.3682, -7.6550,  6.4805,  7.8060]])
```

As we know, the 1st order derivative of such a layer is $-2 \cdot (y - W x )x^T$. One could check with
```python
-torch.outer(y-(torch.matmul(W, x)), x)*2

# tensor([[ 2.3307,  2.4076,  4.2191, -3.5718, -4.3023],
#         [ 3.2671,  3.3749,  5.9143, -5.0069, -6.0309],
#         [-4.2287, -4.3682, -7.6550,  6.4805,  7.8060]], grad_fn=<MulBackward0>)
```

### 2.2. A tensor-train layer

#### 2.2.1 We create a toy dataset and verify the existence of a solution:

```python

n_samples = 256
n_informative = 128

n_input = 4096
n_output = 16

X, y, coef = make_regression(n_samples=n_samples, n_features=n_input, n_informative=n_informative, n_targets=n_output, coef=True)
X = (X - X.mean(0)) / X.std(0)
y = (y - y.mean(0)) / y.std(0)

lr = LinearRegression()
lr.fit(X, y)

lr_y_hat = lr.predict(X)
mean_squared_error(y.reshape(-1), lr_y_hat.reshape(-1)), r2_score(y.reshape(-1), lr_y_hat.reshape(-1))

```

#### 2.2.2. We decompose the input and output dimensions as
$$
    4096 = 8 \times 8 \times 8 \times 8,  \\
    16 = 2 \times 2 \times 2 \times 2
$$
and implement
```python

m = 8
n = 2

X = X.reshape(n_samples, m, m, m, m)
y = y.reshape(n_samples, n, n, n, n)

X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y, dtype=torch.float)
```

#### 2.2.3. Define the rank of the decomposition: 
```python
r = 8
W1_init = np.random.randn(m, n, r)
W1_init /= (W1_init**2).sum()**0.5

W2_init = np.random.randn(m, n, r, r)
W2_init /= (W2_init**2).sum()**0.5

W3_init = np.random.randn(m, n, r, r)
W3_init /= (W3_init**2).sum()**0.5

W4_init = np.random.randn(m, n, r)
W4_init /= (W4_init**2).sum()**0.5

W1 = torch.tensor(W1_init, requires_grad=True, dtype=torch.float)
W2 = torch.tensor(W2_init, requires_grad=True, dtype=torch.float)
W3 = torch.tensor(W3_init, requires_grad=True, dtype=torch.float)
W4 = torch.tensor(W4_init, requires_grad=True, dtype=torch.float)

```

#### 2.2.4. Perform the gradient descent: 
```python
opt = torch.optim.Adam([W1, W2, W3, W4], lr=5e-2)
loss = torch.nn.MSELoss()

n_epochs = 10000

for i in range(n_epochs):
    y_hat = torch.einsum('aei, bfij, cgjk, dhk, nabcd -> nefgh', W1, W2, W3, W4, X)
    error = loss(y_hat.reshape(n_samples, -1), y.reshape(n_samples, -1))
    error.backward()
    opt.step()
    opt.zero_grad()
    
```
Note that we only need a single line to define a tensor-train layer and the `backward()` and `step()` are boilerplate from torch. 

Of course we could have implemented the update from scratch, such as
```python
lr = 5e-2
for i in range(10000): 
    y_hat = torch.einsum('aei, bfij, cgjk, dhk, nabcd -> nefgh', W1, W2, W3, W4, X)
    error = ((y-y_hat)**2).mean()

    error.backward()
    with torch.no_grad():
        W1 -= lr*W1.grad 
        W2 -= lr*W2.grad
        W3 -= lr*W3.grad
        W4 -= lr*W4.grad

        W1.grad.zero_()
        W2.grad.zero_()
        W3.grad.zero_()
        W4.grad.zero_()

```
But at least for this example, `adam` performs way better than plain SGD. 

#### 2.2.5. Evaluation
```python
y_pred = torch.einsum('aei, bfij, cgjk, dhk, nabcd -> nefgh', W1, W2, W3, W4, X).detach().numpy()

print(mean_squared_error(y.numpy().reshape(-1), y_pred.reshape(-1)))

print(r2_score(y.numpy().reshape(-1), y_pred.reshape(-1)))

# 0.055088233
# 0.9449117677629324

```

It cannot fit the data as perfectly as the full model did. But keep in mind that we only use 3.5% of the parameters: 
```python

tt_size = np.prod(W1_init.shape) + np.prod(W2_init.shape) + np.prod(W3_init.shape) + np.prod(W4_init.shape)
full_size = n_input * n_output

tt_size / full_size
# 0.03515625
```