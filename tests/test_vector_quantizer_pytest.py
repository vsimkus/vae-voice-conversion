import pytest

import numpy as np
import torch
from torch import nn

from models.vq_functions import QuantizeVector
from models.vqvae import VectorQuantizer

def test_forward():
    input = torch.rand((2, 3, 5, 7), dtype=torch.float32, requires_grad=True)
    embedding = nn.Embedding(11, 7)
    emb = embedding.weight
    output, latents = QuantizeVector.apply(input, emb)

    differences = input.unsqueeze(3) - emb
    distances = torch.norm(differences, p=2, dim=4)

    _, indices_torch = torch.min(distances, dim=3)

    assert np.allclose(latents.detach().numpy(), indices_torch.detach().numpy()), \
        'Discrete latent encodings are incorrect.'
    assert torch.all(torch.eq(embedding(latents), output)), \
        'Sample from discrete latent (output) is incorrect.'

def test_shape():
    input = torch.rand((2, 3, 5, 7), dtype=torch.float32, requires_grad=True)
    emb = torch.rand((11, 7), dtype=torch.float32, requires_grad=True)
    output, latents = QuantizeVector.apply(input, emb)

    assert output.size() == (2, 3, 5, 7)
    assert output.requires_grad
    assert output.dtype == torch.float32, \
        'Output type does not match input type.'

    assert latents.size() == (2, 3, 5)
    assert not latents.requires_grad
    assert latents.dtype == torch.int64, \
        'Latents are note integer.'

def test_input_straight_through_grad():
    # Straight-through estimator should return the same output grads from output
    input = torch.rand((2, 3, 5, 7), dtype=torch.float32, requires_grad=True)
    emb = torch.rand((11, 7), dtype=torch.float32, requires_grad=True)
    output, _ = QuantizeVector.apply(input, emb)

    grad_output = torch.rand((2, 3, 5, 7))
    grad_input, = torch.autograd.grad(output, 
                                    input,
                                    grad_outputs=[grad_output])

    assert grad_input.size() == (2, 3, 5, 7), \
        'Gradient output shape does not match the input.'
    assert np.allclose(grad_output.numpy(), grad_input.numpy()), \
        'Gradients are not passed straight-through to the input.'

def test_emb_grad():
    # Embedding vector gradient is the same as torch.embedding function
    input = torch.rand((2, 3, 5, 7), dtype=torch.float32, requires_grad=True)
    emb = torch.rand((11, 7), dtype=torch.float32, requires_grad=True)
    output, latents = QuantizeVector.apply(input, emb)

    output_torch = torch.embedding(emb, latents, padding_idx=-1,
        scale_grad_by_freq=False, sparse=False)

    grad_output = torch.rand((2, 3, 5, 7), dtype=torch.float32)
    grad_emb, = torch.autograd.grad(output, emb,
        grad_outputs=[grad_output])
    grad_emb_torch, = torch.autograd.grad(output_torch, emb,
        grad_outputs=[grad_output])

    assert grad_emb.size() == (11, 7), \
        'Embedding gradient shape does not match the embedding space.'
    assert np.allclose(grad_emb.numpy(), grad_emb_torch.numpy()), \
        'Embedding gradients are not equal to the torch.embedding function.'

def test_model_straight_through_grad():
    input = torch.rand((2, 7, 5, 3), dtype=torch.float32, requires_grad=True)
    num_embeddings, embedding_dim = 11, 7
    
    vq = VectorQuantizer(num_embeddings, embedding_dim)
    quantized_sg, _ = vq.forward(input)

    grad_output = torch.rand((2, 7, 5, 3))
    quantized_sg.backward(grad_output)

    assert np.allclose(input.grad.numpy(), grad_output.numpy()), \
        'Gradients are not passed straight-through to the input.'
    assert vq.embedding.weight.grad == None, \
        'Straight-through gradient should not update embeddings.'

def test_model_emb_grad():
    input = torch.rand((2, 7, 5, 3), dtype=torch.float32, requires_grad=True)
    num_embeddings, embedding_dim = 11, 7

    vq = VectorQuantizer(num_embeddings, embedding_dim)
    _, quantized = vq.forward(input)

    grad_output = torch.rand((2, 7, 5, 3))
    quantized.backward(grad_output)

    assert np.any(vq.embedding.weight.grad.numpy()), \
        'Gradient was not used to update the embedding vectors.'
    assert input.grad == None, \
        'Gradient should not be used to update inputs.'