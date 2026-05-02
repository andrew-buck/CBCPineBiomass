import torch
import torch.nn as nn

class ChangePointRegression(nn.Module):
    def __init__(self, K, x_min, x_max):
        super().__init__()

        self.K = K

        # linear terms
        self.beta = nn.Parameter(torch.zeros(K + 2))

        # change-points (initialized evenly)
        if K > 0:
            taus = torch.linspace(x_min, x_max, K + 2)[1:-1]
            self.tau = nn.Parameter(taus)
        else:
            self.tau = None

    def forward(self, x):
        x = x.view(-1, 1)

        eta = self.beta[0] + self.beta[1] * x

        if self.K > 0:
            for k in range(self.K):
                hinge = torch.clamp(x - self.tau[k], min=0.0)
                eta = eta + self.beta[k + 2] * hinge

        return torch.sigmoid(eta)

