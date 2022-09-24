import math
import torch
import torch.nn as nn

def get_freq_indices(method, channel):
    assert method in ['top','bot','low']
    assert channel <= 32
    if 'top' == method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:channel]
        mapper_y = all_top_indices_y[:channel]
    elif 'low' == method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:channel]
        mapper_y = all_low_indices_y[:channel]
    elif 'bot' == method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:channel]
        mapper_y = all_bot_indices_y[:channel]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, freq_sel_method = 'top', activate_type='leaky'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.dct_layer = DCTLayer(channel, dct_h, dct_w, freq_sel_method)

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
    def __init__(self, channel, height, width, freq_sel_method):
        super(DCTLayer, self).__init__()
        mapper_x, mapper_y = get_freq_indices(freq_sel_method, channel)
        mapper_x, mapper_y = [temp_x * (height // 7) for temp_x in mapper_x], [temp_y * (height // 7) for temp_y in mapper_y]
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