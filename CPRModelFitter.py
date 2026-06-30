import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from ChangePointRegression import ChangePointRegression

class CPRModelFitter:
    def __init__(self, x, y, max_K=3, steps=2000):
        self.x = x
        self.y = y
        self.max_K = max_K
        self.steps = steps
        self.best_model = None
        self.results = []

    def __fit_model(self, model, lr=0.05):
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.BCELoss()
        for _ in range(self.steps):
            optimizer.zero_grad()
            p = model(self.x)
            self.y = self.y.unsqueeze(1)
            loss = loss_fn(p, self.y)
            self.y = self.y.squeeze()
            loss.backward()
            optimizer.step()
        return model

    def __bic_score(self, model):
        with torch.no_grad():
            p = model(self.x).clamp(1e-6, 1 - 1e-6)
            loglik = torch.sum(
                self.y * torch.log(p) + (1 - self.y) * torch.log(1 - p)
            )

        k_params = sum(p.numel() for p in model.parameters())
        n = len(self.y)

        bic = -2 * loglik.item() + k_params * np.log(n)
        return bic

    def fit_change_point_model(self):
        x_min, x_max = self.x.min().item(), self.x.max().item()
        best_bic = float("inf")
        self.results = []
        for K in range(self.max_K + 1):
            model = ChangePointRegression(K, x_min, x_max)
            model = self.__fit_model(model)
            bic = self.__bic_score(model)
            self.results.append((K, bic, model))
            if bic < best_bic:
                best_bic = bic
                self.best_model = model

    def plt_model(self):
        x_grid = torch.linspace(self.x.min(), self.x.max(), 300)
        with torch.no_grad():
            p_grid = self.best_model(x_grid)
        #plt.scatter(self.x.numpy(), self.y.numpy(), alpha=0.3)
        plt.plot(x_grid.numpy(), p_grid.numpy(), color="red")
        dividing_line = self.best_model.tau.detach().numpy()[0]
        plt.plot([dividing_line, dividing_line], [0,1], color='black', linestyle='--', linewidth=1)
        plt.xlabel("Rows on the Plot")
        plt.ylabel("P(SP)")
        plt.title("Logistic Regression Change-Point Model")
        plt.text(x=dividing_line + 1, y=0.5, s=f"Change-Point = {dividing_line:.2f}")
        plt.show()