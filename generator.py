import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, ngpu, ni, no):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.n_feature_map = 64
        self.input_size = ni
        self.output_size = no

        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.input_size, self.n_feature_map*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.n_feature_map*8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.n_feature_map*8, self.n_feature_map*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.n_feature_map*4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.n_feature_map*4, self.n_feature_map*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.n_feature_map*2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.n_feature_map*2, self.n_feature_map, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.n_feature_map),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.n_feature_map, self.output_size, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
