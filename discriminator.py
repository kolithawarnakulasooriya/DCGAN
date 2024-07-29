import torch.nn as nn

class Discriminator(nn.Module):

    def __init__(self, ngpu, input_size):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.n_feature_map = 64
        self.input_size = input_size

        self.main = nn.Sequential(
            nn.Conv2d(self.input_size, self.n_feature_map, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.n_feature_map, self.n_feature_map*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.n_feature_map*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.n_feature_map*2, self.n_feature_map*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.n_feature_map*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.n_feature_map*4, self.n_feature_map*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.n_feature_map*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.n_feature_map*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)