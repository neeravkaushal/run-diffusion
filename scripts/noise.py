import torch

class NoiseDealer():
    def __init__(self, beta_start, beta_end, num_timesteps):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1-self.alpha_cum_prod)

    def add_noise(self, x0, noise, T):
        #---T should be a list of size batch size
        bs = x0.shape[0]
        sqrt_alpha_cum_prod_T = self.sqrt_alpha_cum_prod.to(x0.device)[T].reshape(bs)[:,None,None,None]
        sqrt_one_minus_alpha_cum_prod_T = self.sqrt_one_minus_alpha_cum_prod.to(x0.device)[T].reshape(bs)[:,None,None,None]
        xT = sqrt_alpha_cum_prod_T * x0 + sqrt_one_minus_alpha_cum_prod_T * noise.to(x0.device)
        return xT
    
    def sample_image_at_previous_timestep(self, xT, noise_pred, T):
        #---T should be a list of size batch size
        bs = xT.shape[0]
        dev = xT.device

        #print(T)
        #print(self.sqrt_alpha_cum_prod[T].shape)
        sqrt_alpha_cum_prod_T = self.sqrt_alpha_cum_prod[T].reshape(bs)[:,None,None,None].to(dev)

        
        sqrt_one_minus_alpha_cum_prod_T = self.sqrt_one_minus_alpha_cum_prod[T].reshape(bs)[:,None,None,None].to(dev)

        x0 = (xT - (sqrt_one_minus_alpha_cum_prod_T*noise_pred.to(dev))) / sqrt_alpha_cum_prod_T
        x0 = torch.clamp(x0, -1.0, 1.0)

        #print('1 ', self.betas[T].shape)
        #print('2 ', noise_pred.shape)
        #print('3 ', (self.betas[T][:,None,None,None]*noise_pred).shape)
        #print('4 ', xT.shape)
        mean = (xT -  (self.betas[T][:,None,None,None].to(dev)*noise_pred)/(sqrt_one_minus_alpha_cum_prod_T) ) / torch.sqrt(self.alphas[T][:,None,None,None]).to(dev)

        if torch.all(T)==0:
            return mean, x0
        else:
            variance = (self.betas[T][:,None,None,None].to(dev) * (1.0 - self.alpha_cum_prod[T-1][:,None,None,None].to(dev))) / (1.0 - self.alpha_cum_prod[T][:,None,None,None].to(dev))
            std = variance ** 0.5
            #print('5 ', mean.shape)
            #print('6 ', std.shape)
            #print('7 ', xT.shape)
            xT_minus_1 = mean + std * torch.randn(xT.shape).to(dev)
            return xT_minus_1, x0