import torch
import pytorch_lightning as L
from src.models.iaf import IAF

import matplotlib.pyplot as plt
import numpy as np
from src.utils.printarr import printarr

class IAFTrainer(L.LightningModule):

    def __init__(self, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.iaf = IAF(2, [128, 128], n_layers)
    
    def forward(self, x):
        with torch.no_grad():
            z = self.iaf.inverse(x)
        _x, log_det = self.iaf(z)

        # assert torch.allclose(x, _x, atol=1e-5), 'x != _x'
        return z, log_det
    
    def training_step(self, batch, batch_idx):
        x = batch
        z, log_s = self(x)
        nll = -log_s + 0.5 * (z**2).sum(-1) + 0.5 * np.log(2*np.pi)
        loss = nll.mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-5)
    


if __name__ == "__main__":
    # generate data
    x = torch.randn(1000, 2)
    x[:, 1] = x[:, 1] * 2
    x[:, 0] = x[:, 0] + x[:, 1]**2/2
    y = list(x)

    # Dataloader
    x = torch.utils.data.DataLoader(y, batch_size=100, shuffle=True)

    # train
    model = IAFTrainer(5)
    trainer = L.Trainer(max_epochs=500, accelerator='cpu', detect_anomaly=True)
    trainer.fit(model, x)


    with torch.no_grad():
        # plot density
        x = np.linspace(-3, 3, 100)
        y = np.linspace(-3, 3, 100)
        printarr(x, y)
        x, y = np.meshgrid(x, y)
        coords = np.dstack([x, y]).astype(np.float32)

        w = np.zeros_like(coords).astype(np.float32)
        w[..., 1] = coords[..., 1] * 2
        w[..., 0] = coords[..., 0] + coords[..., 1]**2/2

        pdf = np.exp(-0.5 * (coords ** 2).sum(-1) - 0.5 * np.log(2*np.pi))
        plt.subplot(1,2,1)
        plt.contour(w[..., 0], w[..., 1], pdf)
        plt.title('Real')


        w_pred, log_det = model.iaf(torch.from_numpy(coords).reshape(-1, 2))
        w_pred = w_pred.detach().numpy().reshape(coords.shape)

        log_y_pred = -0.5 * (torch.from_numpy((coords ** 2)).sum(-1) + np.log(2*np.pi) + log_det.reshape(coords.shape[:-1]))
        pdf = log_y_pred.exp().detach().numpy()
        printarr(x, y, log_det, log_y_pred, pdf, w_pred)
        plt.subplot(1,2,2)


        plt.contour(w_pred[..., 0], w_pred[..., 1], pdf)
        plt.title('IAF')
        plt.show()