import torch
import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    def forward(self, x):
        return x + self.fn(x)

def ResidualBlock(in_channels, out_channels, kernel_size, dilation):
    return Residual(nn.Sequential(
        nn.BatchNorm1d(in_channels),
        nn.ReLU(),
        nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding='same'),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, padding='same')
    ))

class SpliceAI_80nt(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(4, 32, 1, dilation=1, padding='same')
        self.res_conv1 = nn.Conv1d(32, 32, 1, dilation=1, padding='same')

        self.block1 = nn.Sequential(
            ResidualBlock(32, 32, 11, 1),
            ResidualBlock(32, 32, 11, 1),
            ResidualBlock(32, 32, 11, 1),
            ResidualBlock(32, 32, 11, 1),
            nn.Conv1d(32, 32, 1, dilation=1, padding='same')
        )

        self.conv_last = nn.Conv1d(32, 3, 1, dilation=1, padding='same')

    def forward(self, x):
        x = self.conv1(x)
        detour = self.res_conv1(x)
        x = detour + self.block1(x)
        x = self.conv_last(x)

        return x[..., x.size(0) // 2].softmax(dim=-1)

class SpliceAI_400nt(nn.Module):
    def __init__(self):
        super().__init__()

class SpliceAI_2k(nn.Module):
    def __init__(self):
        super().__init__()

class SpliceAI_10k(nn.Module):
    def __init__(self):
        super().__init__()

class SpliceAI():

    @staticmethod
    def from_preconfigured(model_name):
        if model_name == '80nt':
            return SpliceAI_80nt()
        elif model_name == '400nt':
            return SpliceAI_400nt()
        elif model_name == '2k':
            return SpliceAI_2k()
        elif model_name == '10k':
            return SpliceAI_10k()
        else:
            raise ValueError('Unknown model name: {}'.format(model_name))

if __name__ == '__main__':
    import torch
    x = torch.randn([16, 4, 80])

    model = SpliceAI_80nt()
    print(model(x).shape)