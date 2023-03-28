
import torch 
import torch.nn as nn
import numpy as np

import pytorch_lightning as L
from src.utils.printarr import printarr


class VectorQuantizerST(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs, codebook):
        """
        Args:
            inputs: Tensor, shape [B, C, H, W]
            codebook: Tensor, shape [K, C]
        Returns:
            quantized: Tensor, shape [B, C, H, W]
            quantized_indices: LongTensor, shape [B, H, W]
            diff: Tensor, shape [B, H, W, K]
        """
        B, C, H, W = inputs.shape
        K = codebook.shape[0]

        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        inputs = inputs.view(-1, 1, C)

        distance = torch.sum((inputs - codebook.unsqueeze(0)) ** 2, dim=2) # B x K x C -> B x K
        quantized_indices = torch.argmin(distance, dim=1)
        
        codes = torch.index_select(codebook, 0, quantized_indices)
        ctx.mark_non_differentiable(quantized_indices)
        ctx.save_for_backward(inputs, codebook, quantized_indices)
        
        codes = codes.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        quantized_indices = quantized_indices.view(B, H, W)
        return (codes, quantized_indices)

    @staticmethod
    def backward(ctx, grad_codes, grad_indices):
        inputs, codebook, indices = ctx.saved_tensors
        B, C, H, W = grad_codes.shape
        K = codebook.shape[0]

        grad_inputs = grad_codebook = None

        if ctx.needs_input_grad[0]:
            # Straight-through gradient
            grad_inputs = grad_codes.clone()

        if ctx.needs_input_grad[1]:
            # gradient of the codebook
            embedding_size = codebook.size(1)
            
            grad_codes = grad_codes.permute(0, 2, 3, 1).contiguous()
            grad_output_flatten = (grad_codes.contiguous()
                                              .view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)


        return (grad_inputs, grad_codebook)
    
class VectorQuantizer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs, codebook):
        """
        Args:
            inputs: Tensor, shape [B, C, H, W]
            codebook: Tensor, shape [K, C]
        Returns:
            quantized: Tensor, shape [B, C, H, W]
            quantized_indices: LongTensor, shape [B, H, W]
            diff: Tensor, shape [B, H, W, K]
        """
        B, C, H, W = inputs.shape
        K = codebook.shape[0]

        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        inputs = inputs.view(-1, 1, C)
        distance = torch.sum((inputs - codebook.unsqueeze(0)) ** 2, dim=2) # B x K x C -> B x K
        quantized_indices = torch.argmin(distance, dim=1)
        
        codes = torch.index_select(codebook, 0, quantized_indices)
        ctx.mark_non_differentiable(quantized_indices)
        ctx.save_for_backward(inputs, codebook, quantized_indices)
        
        codes = codes.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        quantized_indices = quantized_indices.view(B, H, W)
        print('VQ not ST')
        printarr(codes)
        return (codes, quantized_indices)

    @staticmethod
    def backward(ctx, grad_codes, grad_indices):
        raise NotImplementedError('VectorQuantizer does not support backward pass')



class Quantizer(nn.Module):
    def __init__(self, codebook_size=10, embedding_dim=10):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self._codebook = torch.rand(codebook_size, embedding_dim)
        self.codebook = nn.Parameter(self._codebook)


    def forward(self, z):
        return VectorQuantizerST.apply(z, self.codebook)


class VQVAE(L.LightningModule):
    def __init__(self, encoder, decoder, codebook_size=10, embedding_dim=32, commitment_const=0.25, lr=1e-3):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.commitment_const = commitment_const
        self.lr = lr
        self.quantizer = Quantizer(codebook_size=codebook_size, embedding_dim=embedding_dim)
    
    def forward(self, x):
        z = self.encoder(x)
        z_q, _  = self.quantizer(z)
        x_recon = self.decoder(z_q)
        return x_recon, z, z_q
    
    def loss(self, x, x_recon, z, z_q):
        # printarr(z, z_q, x, x_recon)
        B = x.shape[0]
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction='none').reshape(B, -1).sum(-1)
        quantization_loss = nn.functional.mse_loss(z_q.detach(), z, reduction='none').reshape(B, -1).sum(-1)
        commitment_loss = nn.functional.mse_loss(z_q, z.detach(), reduction='none').reshape(B, -1).sum(-1)
        # printarr(recon_loss, commitment_loss, quantization_loss)
        
        return recon_loss, quantization_loss, commitment_loss

    def _run_step(self, batch, batch_idx):
        x, y = batch
        x_recon, z, z_q = self(x)
        loss = self.loss(x, x_recon, z, z_q)
        return loss

    def training_step(self, batch, batch_idx):
        recon_loss, quantization_loss, commitment_loss = self._run_step(batch, batch_idx)
        loss = (recon_loss  + quantization_loss + self.commitment_const * commitment_loss).mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        recon_loss, _, _  = self._run_step(batch, batch_idx)
        recon_loss = recon_loss.mean()
        self.log('val_loss', recon_loss, on_epoch=True, prog_bar=True, logger=True)
        return recon_loss
    
    def test_step(self, batch, batch_idx):
        recon_loss, _, _  = self._run_step(batch, batch_idx)
        recon_loss = recon_loss.mean()
        self.log('test_loss', recon_loss)
        return recon_loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    

if __name__=='__main__':

    vq_st = VectorQuantizerST.apply
    vq = VectorQuantizer.apply

    def test_vq_st_gradient1():
        inputs = torch.rand((2, 7, 5, 3), dtype=torch.float32, requires_grad=True)
        codebook = torch.rand((11, 7), dtype=torch.float32, requires_grad=True)
        codes, _ = vq_st(inputs, codebook)

        grad_output = torch.rand((2, 7, 5, 3))
        grad_inputs, = torch.autograd.grad(codes, inputs,
            grad_outputs=[grad_output])

        # Straight-through estimator
        assert grad_inputs.size() == (2, 7, 5, 3)
        assert np.allclose(grad_output.numpy(), grad_inputs.numpy())

    def test_vq_st_gradient2():
        inputs = torch.rand((2, 7, 5, 3), dtype=torch.float32, requires_grad=True)
        codebook = torch.rand((11, 7), dtype=torch.float32, requires_grad=True)
        codes, indices = vq_st(inputs, codebook)
        # codes = codes.permute(0,2,3,1).contiguous()
        _, indices = vq(inputs, codebook)
        codes_torch = torch.embedding(codebook, indices.reshape(-1, 1), padding_idx=-1,
            scale_grad_by_freq=False, sparse=False).reshape((2, 5, 3, 7)).permute(0, 3, 1, 2).contiguous()
        printarr(codes, indices, codes_torch)
        grad_output = torch.rand((2, 7, 5, 3), dtype=torch.float32)
        grad_codebook, = torch.autograd.grad(codes, codebook,
            grad_outputs=[grad_output])
        grad_codebook_torch, = torch.autograd.grad(codes_torch, codebook,
            grad_outputs=[grad_output])

        # Gradient is the same as torch.embedding function
        assert grad_codebook.size() == (11, 7)
        assert np.allclose(grad_codebook.numpy(), grad_codebook_torch.numpy())

    test_vq_st_gradient1()
    test_vq_st_gradient2()