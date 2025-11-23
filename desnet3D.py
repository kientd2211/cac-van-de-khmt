import torch
import torch.nn as nn
import torch.nn.functional as F


class _DenseLayer3D(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        inter_channels = bn_size * growth_rate
        self.norm1 = nn.BatchNorm3d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, inter_channels, kernel_size=1, bias=False)

        self.norm2 = nn.BatchNorm3d(inter_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)

        self.drop_rate = float(drop_rate)

    def forward(self, x):
        out = self.conv1(self.relu1(self.norm1(x)))
        out = self.conv2(self.relu2(self.norm2(out)))
        if self.drop_rate > 0:
            out = F.dropout3d(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)


class _DenseBlock3D(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        layers = []
        channels = in_channels
        for i in range(num_layers):
            layer = _DenseLayer3D(channels, growth_rate, bn_size, drop_rate)
            layers.append(layer)
            channels += growth_rate
        self.layers = nn.ModuleList(layers)
        self.out_channels = channels

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _Transition3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(self.relu(self.norm(x)))
        x = self.pool(x)
        return x


class DenseNet3D(nn.Module):
    """
    Configurable 3D DenseNet.

    Default config here is small (suitable for small patches like 32^3).
    You can create larger variants via helper factory functions below.
    """

    def __init__(self, in_channels=1, growth_rate=12, block_config=(4, 4, 4),
                 num_init_features=24, bn_size=4, drop_rate=0.0, num_classes=1):
        super().__init__()

        # Initial convolution
        self.features = nn.Sequential()
        self.features.add_module('conv0', nn.Conv3d(in_channels, num_init_features,
                                                   kernel_size=3, stride=1, padding=1, bias=False))
        self.features.add_module('norm0', nn.BatchNorm3d(num_init_features))
        self.features.add_module('relu0', nn.ReLU(inplace=True))

        num_features = num_init_features

        # DenseBlocks + Transitions
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock3D(num_layers=num_layers, in_channels=num_features,
                                  growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = block.out_channels

            # Do not add transition after last block
            if i != len(block_config) - 1:
                trans_out = num_features // 2
                trans = _Transition3D(in_channels=num_features, out_channels=trans_out)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = trans_out

        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm3d(num_features))

        # Global average pooling and classifier head (output logits)
        self.classifier = nn.Linear(num_features, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (B, C=1, D, H, W)
        features = self.features(x)
        out = F.relu(features)
        # Global average pool to (B, num_features, 1,1,1)
        out = F.adaptive_avg_pool3d(out, (1, 1, 1)).view(features.size(0), -1)
        logits = self.classifier(out)
        return logits

def densenet3d121(in_channels=1, num_classes=1):
    # larger model roughly analogous to DenseNet-121 but in 3D
    return DenseNet3D(in_channels=in_channels, growth_rate=32, block_config=(6, 12, 24, 16),
                       num_init_features=64, bn_size=4, drop_rate=0.0, num_classes=num_classes)
