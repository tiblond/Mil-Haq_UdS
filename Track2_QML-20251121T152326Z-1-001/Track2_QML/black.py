import torch as torch
import torch.nn as nn
from torch.distributions import Normal

class DifferentiableBlackModel(nn.Module):
    def __init__(self):
        super.__init__()

    def forward(self, f, k, sigma, t, r, type='call'):
        """
        f : future price
        r : interest rate
        sigma : volatility
        t : maturity
        k : strike price
        type : either call (0) or put(1)
        """

        if not isinstance(f, torch.Tensor): f = torch.tensor(f)
        if not isinstance(sigma, torch.Tensor): sigma = torch.tensor(sigma)

        d1 = (torch.log(f / k) + ((sigma**2 / 2)*t)) / (sigma * torch.sqrt(t))
        d2 = (torch.log(f / k) - ((sigma**2 / 2)*t)) / (sigma * torch.sqrt(t))

        sigma = torch.clamp(sigma, min=1e-6)
        t = torch.clamp(torch.tensor(t, device=f.device), min=1e-6)
        
        norm_distribution = Normal(0, 1)

        if type == "0":
            p = torch.exp(-r * t) * (f * norm_distribution.cdf(d1) - k * norm_distribution.cdf(d2))
        else:
            p = torch.exp(-r * t) * (k * norm_distribution.cdf(-d2) - f * norm_distribution.cdf(-d1))

        return p
    
class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred_params, true_price, static_params, scaler_mean, scaler_std):
        """
        pred_params: Tensor [Batch, 2] (Predicted Normalized F, Normalized Sigma)
        true_price:  Tensor [Batch, 1] (Actual Market Prices)
        """

        mean_t = torch.tensor(scaler_mean, device=pred_params.device).float()
        std_t = torch.tensor(scaler_std, device=pred_params.device).float()
        
        real_vals = (pred_params * std_t) + mean_t
        
        pred_F = real_vals[:, 0]
        pred_sigma = torch.abs(real_vals[:, 1]) 

        pred_price = DifferentiableBlackModel.forward(
            f=pred_F,
            k=static_params['k'],
            sigma=pred_sigma,
            t=static_params['t'],
            r=static_params['r'],
        )
        
        return self.mse(pred_price, true_price.squeeze())


