import torch
from torch.autograd import Function

class QuantizeVector(Function):
    """
    Implements the algorithm presented in
    'Neural Discrete Representation Learning' by van den Oord et al.
    https://arxiv.org/abs/1711.00937

    Quantized tensor into a discrete representation. The last dimension is used for discretization.
    Returns discrete representation, and reconstructed representation from nearest neighbours.

    The output tensor will have the same shape as the input.
    The discrete tensor will have the same shape except for last dimension which is used for discretization.

    Code based on:
    https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py
    https://github.com/ritheshkumar95/pytorch-vqvae/blob/master/functions.py
    """
    @staticmethod
    def forward(ctx, input, emb):
        with torch.no_grad():
            embedding_dim = emb.size(1)
            input_size = input.size()

            # Number of channels must match the embedding dimension
            if input.size(-1) != embedding_dim:
                raise ValueError('Input channels must match ({}) \
                                    must match embedding dimension ({})'
                                    .format(input.size(-1), embedding_dim))
            
            flat_input = input.view(-1, embedding_dim)

            # Compute the Approximate Nearest Neighbours to the embeddings (eq. 2 from VQ-VAE paper)
            emb_sqr = torch.sum(emb ** 2, dim=1)
            input_sqr = torch.sum(flat_input ** 2, dim=1, keepdim=True)
            distance = torch.addmm(emb_sqr + input_sqr,
                                flat_input, 
                                emb.t(), 
                                alpha=-2.0, 
                                beta=1.0)

            # Flattened latent vector found by minimizing distance
            _, flat_latents = torch.min(distance, dim=1)
            latents = flat_latents.view(*input_size[:-1])
            
            ctx.save_for_backward(flat_latents, emb)
            ctx.mark_non_differentiable(latents)

        # TODO: Watchout and test, this not sure if this is right.
        # But I think the we need gradients for the below tensors

        # Sample from the embeddings using the latent vector
        # to construct input tensor for the decoder
        # TODO: Use Embedding class instead
        flat_output = torch.index_select(emb, 
                                        dim=0,
                                        index=flat_latents)
        output = flat_output.view(input_size)

        return output, latents
    
    @staticmethod
    def backward(ctx, grad_output, grad_latents):
        grad_input, grad_emb = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator for the quantizer
            grad_input = grad_output.clone()
        
        if ctx.needs_input_grad[1]:
            # Gradient wrt. embeddings
            flat_latents, emb = ctx.saved_tensors
            embedding_dim = emb.size(1)

            flat_grad_output = (grad_output.view(-1, embedding_dim))
            grad_emb= torch.zeros_like(emb)
            grad_emb.index_add_(0, flat_latents, flat_grad_output)
        
        return grad_input, grad_emb
        