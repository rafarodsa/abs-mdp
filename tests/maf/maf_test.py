import torch
import pytorch_lightning as L
from src.models.maf import MAF

import matplotlib.pyplot as plt
import numpy as np
from src.utils.printarr import printarr

class MAFTrainer(L.LightningModule):

    def __init__(self, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.maf = MAF(2, [128,128,128], n_layers)
    
    def forward(self, x):
        z, log_det = self.maf(x)
        return z, log_det
    
    def training_step(self, batch, batch_idx):
        x = batch
        z, log_s = self(x)
        nll = -(-log_s - 0.5 * (z**2).sum(-1) - 0.5 * np.log(2*np.pi)) # assuming base distribution u ~ N(0, I)
        loss = nll.mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    


if __name__ == "__main__":
    # generate data
    x = torch.randn(10000, 2)
    _y = torch.zeros_like(x)
    _y[:, 1] = x[:, 1] * 2
    _y[:, 0] = x[:, 0] + _y[:, 1]**2/2
    log_pdf_y = -np.log(2*np.pi) - 0.5 * (_y[:, 1]/2) ** 2 - 0.5 * (_y[:, 0] - _y[:, 1]**2/2) ** 2
    print(f'Avg NLL of samples: {-log_pdf_y.mean()}')
    y = list(_y)

    # Dataloader
    data = torch.utils.data.DataLoader(y, batch_size=100, shuffle=True)

    # train
    model = MAFTrainer(5)
    trainer = L.Trainer(max_epochs=5, accelerator='cpu', detect_anomaly=True)
    trainer.fit(model, data)

    y_pred, log_s = model.maf.inverse(x)
    y_pred = y_pred.detach().numpy()


    plt.subplot(1,3,1)
    plt.scatter(x[:, 0], x[:, 1], marker='o')
    plt.title('Base samples')
    plt.subplot(1,3,2)
    plt.scatter(_y[:, 0], _y[:, 1], marker='o')
    plt.title('Transformed samples')
    plt.subplot(1,3,3)
    plt.scatter(y_pred[:, 0], y_pred[:, 1], marker='o')
    plt.title('MAF Transformed samples')
    # plt.show()

    plt.figure()
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


        w_pred, log_s = model.maf.inverse(torch.from_numpy(coords).reshape(-1, 2))
        w_pred = w_pred.detach().numpy().reshape(coords.shape)

        log_y_pred = -0.5 * (torch.from_numpy((coords ** 2)).sum(-1) + np.log(2*np.pi)) + log_s.reshape(coords.shape[:-1])
        pdf = log_y_pred.exp().detach().numpy()
        printarr(x, y, log_s, log_y_pred, pdf, w_pred, w)
        plt.subplot(1,2,2)


        plt.contour(w_pred[..., 0], w_pred[..., 1], pdf)
        plt.title('MAF')

        # plt.figure()
        # plt.scatter(w[..., 0], w[..., 1], marker='.')
        # plt.scatter(w_pred[..., 0], w_pred[..., 1], marker='+')


        plt.savefig('maf.png')