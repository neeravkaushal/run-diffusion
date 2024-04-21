import torch
import torch.nn as nn
from  utils import get_sinusoidal_embeddings

class DownBlock(nn.Module):

    def __init__(self, inchn, outchn, time_emb_dim, num_groups, down_sample=True, num_heads=8, num_layers=1):
        super().__init__()
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.down_sample = down_sample

        #----FIRST RESNET BLOCK
        resnet_blocks = []
        for i in range(self.num_layers):
            resnet_block = nn.Sequential(
                                nn.GroupNorm(self.num_groups, inchn if i==0 else outchn),
                                nn.SiLU(),
                                nn.Conv2d(inchn if i == 0 else outchn,
                                          outchn,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1),)
            resnet_blocks.append(resnet_block)
        self.resnet_conv_1 = nn.ModuleList(resnet_blocks)
        del resnet_block, resnet_blocks
        
        #----TIME EMBEDDINGS
        time_emb_blocks = []
        for _ in range(self.num_layers):
            time_emb_block = nn.Sequential(
                                nn.SiLU(),
                                nn.Linear(time_emb_dim, outchn))
            time_emb_blocks.append(time_emb_block)
        self.time_emb_layers = nn.ModuleList(time_emb_blocks)
        del time_emb_block, time_emb_blocks

        #----SECOND RESNET BLOCK
        resnet_blocks = []
        for _ in range(self.num_layers):
            resnet_block = nn.Sequential(
                                nn.GroupNorm(self.num_groups, outchn),
                                nn.SiLU(),
                                nn.Conv2d(outchn,
                                          outchn,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1),)
            resnet_blocks.append(resnet_block)
        self.resnet_conv_2 = nn.ModuleList(resnet_blocks)
        del resnet_block, resnet_blocks


        self.attention_norms = nn.ModuleList([nn.GroupNorm(self.num_groups, outchn) for _ in range(self.num_layers)])
        
        #---ATTENTION BLOCKS
        self.attentions = nn.ModuleList([nn.MultiheadAttention(outchn, self.num_heads, batch_first=True) for _ in range(self.num_layers)])

        #---RESIDUAL BLOCKS
        self.residual_input_conv = nn.ModuleList([nn.Conv2d(inchn if i==0 else outchn, outchn, kernel_size=1) for i in range(self.num_layers)])
        
        #---DOWN SAMPLING CONVOLUTION
        self.down_sample_conv = nn.Conv2d(outchn, outchn, 4, 2, 1) if self.down_sample else nn.Identity()


    def forward(self, x, time_emb):
        out = x
        
        for i in range(self.num_layers):
            
            #---RESNET
            resnet_input = out #---keep for within-resnet residual connection
            out = self.resnet_conv_1[i](out)
            out = out + self.time_emb_layers[i](time_emb)[:, :, None, None] #-- [BS, emb_dim] --> [BS, emb_dim, 1, 1] to match [BS, C, H, W]
            out = self.resnet_conv_2[i](out)
            out = out + self.residual_input_conv[i](resnet_input)
            
            #---ATTENTION
            batch_size, c, h, w = out.shape
            in_attn = out.reshape(batch_size, c, h * w)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, c, h, w)
            
            #---ADD ATTENTION TO RESNET
            out = out + out_attn
            
        #---DOWNSAMPLING
        out = self.down_sample_conv(out)
        
        return out
    

class MidBlock(nn.Module):

    def __init__(self, inchn, outchn, time_emb_dim, num_groups, num_heads=4, num_layers=1):
        super().__init__()

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_groups = num_groups

        #----FIRST RESNET BLOCK
        resnet_blocks = []
        for i in range(self.num_layers+1):
            resnet_block = nn.Sequential(
                                nn.GroupNorm(self.num_groups, inchn if i==0 else outchn),
                                nn.SiLU(),
                                nn.Conv2d(inchn if i == 0 else outchn,
                                          outchn,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1),)
            resnet_blocks.append(resnet_block)
        self.resnet_conv_1 = nn.ModuleList(resnet_blocks)
        del resnet_block, resnet_blocks

        #----TIME EMBEDDINGS
        time_emb_blocks = []
        for _ in range(self.num_layers+1):
            time_emb_block = nn.Sequential(
                                nn.SiLU(),
                                nn.Linear(time_emb_dim, outchn))
            time_emb_blocks.append(time_emb_block)
        self.time_emb_layers = nn.ModuleList(time_emb_blocks)
        del time_emb_block, time_emb_blocks

        #----SECOND RESNET BLOCK
        resnet_blocks = []
        for _ in range(self.num_layers+1):
            resnet_block = nn.Sequential(
                                nn.GroupNorm(self.num_groups, outchn),
                                nn.SiLU(),
                                nn.Conv2d(outchn,
                                          outchn,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1),)
            resnet_blocks.append(resnet_block)
        self.resnet_conv_2 = nn.ModuleList(resnet_blocks)
        del resnet_block, resnet_blocks


        
        self.attention_norms = nn.ModuleList([nn.GroupNorm(self.num_groups, outchn) for _ in range(self.num_layers)])
        
        #---ATTENTION BLOCKS
        self.attentions = nn.ModuleList([nn.MultiheadAttention(outchn, self.num_heads, batch_first=True) for _ in range(self.num_layers)])

        #---RESIDUAL BLOCKS
        self.residual_input_conv = nn.ModuleList([nn.Conv2d(inchn if i==0 else outchn, outchn, kernel_size=1) for i in range(self.num_layers+1)])


    def forward(self, x, time_emb):

        out = x
        
        # First resnet block
        resnet_input = out #---keep for within-resnet residual connection
        out = self.resnet_conv_1[0](out)
        out = out + self.time_emb_layers[0](time_emb)[:, :, None, None] #-- [BS, emb_dim] --> [BS, emb_dim, 1, 1] to match [BS, C, H, W]
        out = self.resnet_conv_2[0](out)
        out = out + self.residual_input_conv[0](resnet_input)
        
        for i in range(self.num_layers):

            #---ATTENTION
            batch_size, c, h, w = out.shape
            in_attn = out.reshape(batch_size, c, h * w)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, c, h, w)

            #---ADD ATTENTION TO RESNET
            out = out + out_attn
            
            # Resnet Block
            resnet_input = out
            out = self.resnet_conv_1[i+1](out)
            out = out + self.time_emb_layers[i+1](time_emb)[:, :, None, None]
            out = self.resnet_conv_2[i+1](out)
            out = out + self.residual_input_conv[i+1](resnet_input)
        
        return out
    

class UpBlock(nn.Module):
    def __init__(self, inchn, outchn, time_emb_dim, up_sample=True, num_heads=4, num_groups=3, num_layers=1):
        super().__init__()

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.up_sample = up_sample
        
        #----FIRST RESNET BLOCK
        resnet_blocks = []
        for i in range(self.num_layers):
            resnet_block = nn.Sequential(
                                nn.GroupNorm(self.num_groups, inchn if i==0 else outchn),
                                nn.SiLU(),
                                nn.Conv2d(inchn if i == 0 else outchn,
                                          outchn,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1),)
            resnet_blocks.append(resnet_block)
        self.resnet_conv_1 = nn.ModuleList(resnet_blocks)
        del resnet_block, resnet_blocks
        
        #----TIME EMBEDDINGS
        time_emb_blocks = []
        for _ in range(self.num_layers):
            time_emb_block = nn.Sequential(
                                nn.SiLU(),
                                nn.Linear(time_emb_dim, outchn))
            time_emb_blocks.append(time_emb_block)
        self.time_emb_layers = nn.ModuleList(time_emb_blocks)
        del time_emb_block, time_emb_blocks

        #----SECOND RESNET BLOCK
        resnet_blocks = []
        for _ in range(self.num_layers):
            resnet_block = nn.Sequential(
                                nn.GroupNorm(self.num_groups, outchn),
                                nn.SiLU(),
                                nn.Conv2d(outchn,
                                          outchn,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1),)
            resnet_blocks.append(resnet_block)
        self.resnet_conv_2 = nn.ModuleList(resnet_blocks)
        del resnet_block, resnet_blocks
        
        self.attention_norms = nn.ModuleList([nn.GroupNorm(self.num_groups, outchn) for _ in range(self.num_layers)])
        
        #---ATTENTION BLOCKS
        self.attentions = nn.ModuleList([nn.MultiheadAttention(outchn, self.num_heads, batch_first=True) for _ in range(self.num_layers)])

        #---RESIDUAL BLOCKS
        self.residual_input_conv = nn.ModuleList([nn.Conv2d(inchn if i==0 else outchn, outchn, kernel_size=1) for i in range(self.num_layers)])

        #---UP SAMPLING CONVOLUTION
        self.up_sample_conv = nn.ConvTranspose2d(inchn//2, inchn//2, 4, 2, 1) if self.up_sample else nn.Identity()
    
    def forward(self, x, out_down, time_emb):
        x = self.up_sample_conv(x)
        x = torch.cat([x, out_down], dim=1)
        
        out = x
        for i in range(self.num_layers):

            #---RESNET
            resnet_input = out #---keep for within-resnet residual connection
            out = self.resnet_conv_1[i](out)
            out = out + self.time_emb_layers[i](time_emb)[:, :, None, None] #-- [BS, emb_dim] --> [BS, emb_dim, 1, 1] to match [BS, C, H, W]
            out = self.resnet_conv_2[i](out)
            out = out + self.residual_input_conv[i](resnet_input)
            
            #---ATTENTION
            batch_size, c, h, w = out.shape
            in_attn = out.reshape(batch_size, c, h * w)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, c, h, w)

            #---ADD ATTENTION TO RESNET
            out = out + out_attn

        return out



class Unet(nn.Module):

    def __init__(self, model_config):
        super().__init__()
        im_channels = model_config['im_channels']
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.time_emb_dim = model_config['time_emb_dim']
        self.down_sample = model_config['down_sample']
        self.up_sample = list(reversed(self.down_sample))
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']
        self.num_groups = model_config['num_groups']
        
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-2]
        assert len(self.down_sample) == len(self.down_channels) - 1
        
        # Initial projection from sinusoidal time embedding
        self.time_proj = nn.Sequential(
                            nn.Linear(self.time_emb_dim, self.time_emb_dim),
                            nn.SiLU(),
                            nn.Linear(self.time_emb_dim, self.time_emb_dim)
        )

        self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=(1, 1))
        
        self.downs = nn.ModuleList([])
        for i in range(len(self.down_channels)-1):
            self.downs.append(DownBlock(self.down_channels[i],
                                        self.down_channels[i+1],
                                        self.time_emb_dim,
                                        down_sample=self.down_sample[i],
                                        num_layers=self.num_down_layers,
                                        num_groups=self.num_groups))
        
        self.mids = nn.ModuleList([])
        for i in range(len(self.mid_channels)-1):
            self.mids.append(MidBlock(self.mid_channels[i],
                                      self.mid_channels[i+1],
                                      self.time_emb_dim,
                                      num_layers=self.num_mid_layers,
                                      num_groups=self.num_groups))
        
        self.ups = nn.ModuleList([])
        for i in reversed(range(len(self.down_channels)-1)):
            self.ups.append(UpBlock(self.down_channels[i]*2,
                                    self.down_channels[i-1] if i != 0 else 16,
                                    self.time_emb_dim,
                                    up_sample=self.down_sample[i],
                                    num_layers=self.num_up_layers,
                                    num_groups=self.num_groups))
        
        self.norm_out = nn.GroupNorm(8, 16)
        self.conv_out = nn.Conv2d(16, im_channels, kernel_size=3, padding=1)
    
    def forward(self, x, t):
        # Shapes assuming downblocks are [C1, C2, C3, C4]
        # Shapes assuming midblocks are [C4, C4, C3]
        # Shapes assuming downsamples are [True, True, False]
        # B x C x H x W
        out = self.conv_in(x)
        # B x C1 x H x W
        
        # time_emb of shape (B x time_emb_dim)
        time_emb = get_sinusoidal_embeddings(torch.as_tensor(t).long(), self.time_emb_dim)
        time_emb = self.time_proj(time_emb)
        
        down_outs = []
        for idx, down in enumerate(self.downs):
            down_outs.append(out)
            out = down(out, time_emb)
        # down_outs:--[B x C1 x H x W,
        #              B x C2 x H/2 x W/2,
        #              B x C3 x H/4 x W/4,
        # out:-------- B x C4 x H/4 x W/4
        #
            
        for mid in self.mids:
            out = mid(out, time_emb)
        # out:-------- B x C3 x H/4 x W/4
        
        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, time_emb)
            # out:---- [B x C2 x H/4 x W/4,
            #           B x C1 x H/2 x W/2,
            #           B x 16 x H x W]
        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out) #----output -----> B x C x H x W
        return out

