import torch
import torch.nn as nn
import torch.nn.functional as F

isprint = False

def pprint(*args, **kwargs):
    if isprint:
        print(*args, **kwargs)

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=4):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.5)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.5)
        self.relu2 = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.se(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x = self.relu2(x + residual)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.5)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class StatisticsPooling(nn.Module):
    def __init__(self, input_dim, stddev=True, eps=1e-9):
        super(StatisticsPooling, self).__init__()
        self.input_dim = input_dim
        self.stddev = stddev
        self.eps=1e-9

        if self.stddev:
            self.output_dim = self.input_dim * 2
        else:
            self.output_dim = self.input_dim

    def forward(self, inputs):

        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        lengths = inputs.shape[2]

        mean = inputs.sum(dim=2, keepdim=True) / lengths

        if self.stddev:
            std = torch.sqrt(
                torch.div(
                    torch.sum((inputs - mean) ** 2, dim=2, keepdim=True), lengths
                ).clamp(min=self.eps)
            )
            mean = mean.squeeze(2)
            std = std.squeeze(2)
            return torch.cat((mean, std), dim=1)

        mean = mean.squeeze(2)
        return mean

class ResNetSE(nn.Module):
    def __init__(self, block, layers, num_filters, input_dim, embedding_size, **kwargs):
        super(ResNetSE, self).__init__()

        self.inplanes = num_filters[0]

        self.conv1 = nn.Conv2d(1, num_filters[0] , kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0], momentum=0.5)
        self.relu = nn.ReLU(inplace=True)


        self.layer1 = self._make_layer(block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=(2, 2))

        stats_input_dim = (((input_dim + 1) // 2 + 1) // 2 + 1) // 2 * num_filters[-1]
        self.pooling = StatisticsPooling(input_dim=stats_input_dim)

        self.fc = nn.Linear(self.pooling.output_dim, embedding_size)
        self.fc_bn = nn.BatchNorm1d(embedding_size, momentum=0.5, affine=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.5),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, lengths=None):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        #x = x.reshape(x.size()[0],-1,x.size()[-1])
        x = x.transpose(-1, -2)
        x = x.reshape(x.shape[0], -1, x.shape[3])

        x = self.pooling(x)

        x = x.view(x.size()[0], -1)

        embedding = self.fc_bn(self.fc(x))


        return embedding

def create_ResNet34Tiny(
    input_dim=80,
    embedding_size=128
):
    return ResNetSE(
        block=SEBasicBlock,
        layers=[3, 4, 6, 3],
        num_filters=[32, 64, 128, 256],
        input_dim=input_dim,
        embedding_size=embedding_size,
    )

if __name__ == "__main__":
    isprint = True
    model = ResNetSE(
        block=SEBasicBlock,
        layers=[3, 4, 6, 3],
        num_filters=[32, 64, 128, 256],
        input_dim=80,
        embedding_size=128
    )
    print(model)
    print("total parameters: {}M".format(sum([p.nelement() for p in model.parameters() if p.requires_grad])/1000/1000))
    inputs = torch.randn((16, 200, 80))
    outputs = model(inputs)
    print(outputs.shape)
