import torch
import torch.nn as nn


class SqueezeLayer(nn.Module):
    def __init__(self, in_channel=256):
        super(SqueezeLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # global average pooling
        self.fc = nn.Sequential(
            nn.Linear(in_channel, 64, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1024, bias=False)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # squeeze
        print(y.shape)
        y = self.fc(y).view(b, 1024)  # squeeze
        return y


class ExcitationLayer(nn.Module):
    def __init__(self, in_channel=1024, out_channel=256):
        super(ExcitationLayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channel, 64, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        b, out_channel, _, _ = x.size()
        y = self.fc(y).view(b, out_channel, 1, 1)

        return x * y.expand_as(x)


if __name__ == "__main__":
    x = torch.randn(1, 256, 16, 10)
    squeezenet = SqueezeLayer(in_channel=256)
    excitationnet = ExcitationLayer(in_channel=1024, out_channel=256)
    y = squeezenet(x)
    print(y.shape)
    out = excitationnet(x, y)
    print(out.shape)

# torch.Size([1, 256])
# torch.Size([1, 1024])
# torch.Size([1, 256, 16, 10])
