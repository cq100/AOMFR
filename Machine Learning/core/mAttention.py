import math
import torch
import torch.nn as nn

def get_freq_indices(channel):
    assert channel <= 15
    all_top_indices_x = [0,0,0,1,1,3,0,2,3,2,3,3,2,2,1]
    all_top_indices_y = [0,1,2,0,2,0,3,2,3,1,2,1,3,0,3]
    mapper_x = all_top_indices_x[:channel]
    mapper_y = all_top_indices_y[:channel]
    return mapper_x, mapper_y

class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w,  activate_type='leaky'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.dct_layer = DCTLayer(channel, dct_h, dct_w)

        if activate_type == "leaky":
            activate = nn.LeakyReLU(0.1)
        elif activate_type == "silu":
            activate = nn.SiLU()
        elif activate_type == "mish":
            activate = nn.Mish()
        else:
            raise ValueError("activate_type is no exis !")
            
        self.fc = nn.Sequential(
            nn.Linear(channel, channel, bias=False),
            activate,
            nn.Linear(channel, channel, bias=False),
            nn.Sigmoid() )

    def forward(self, x):
        att = self.dct_layer(x)
        att = self.fc(att).view(x.shape[0], x.shape[1], 1, 1)
        return x * att.expand_as(x)


class DCTLayer(nn.Module):
    def __init__(self, channel, height, width):
        super(DCTLayer, self).__init__()
        mapper_x, mapper_y = get_freq_indices(channel)
        mapper_x, mapper_y = [temp_x * (height // 4) for temp_x in mapper_x], [temp_y * (height // 4) for temp_y in mapper_y]
        self.num_split = len(mapper_x)
        self.register_buffer('Spectralweight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        
    def forward(self, x):
        return torch.sum(x * self.Spectralweight, dim=[2,3])

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
        if freq == 0: return result
        else: return result * math.sqrt(2)
    
    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)
        c_part = channel // len(mapper_x)
        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i+1)*c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
        return dct_filter