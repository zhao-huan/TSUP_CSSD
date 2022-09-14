import torch
import torch.nn as nn

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

class BLSTMP(nn.Module):
    def __init__(self, n_in, n_hidden, nproj=160, dropout=0, num_layers=1):
        super(BLSTMP, self).__init__()

        self.num_layers = num_layers

        self.rnns = nn.ModuleList([nn.LSTM(n_in, n_hidden, bidirectional=True, dropout=dropout, batch_first=True)])
        self.linears = nn.ModuleList([nn.Linear(2*n_hidden, 2*nproj)])

        for i in range(num_layers-1):
            self.rnns.append(nn.LSTM(2*nproj, n_hidden, bidirectional=True, dropout=dropout, batch_first=True))
            self.linears.append(nn.Linear(2*n_hidden, 2*nproj))

    def forward(self, feature):
        recurrent, _ = self.rnns[0](feature)
        output = self.linears[0](recurrent)

        for i in range(self.num_layers-1):
            output, _ = self.rnns[i+1](output)
            output = self.linears[i+1](output)

        return output

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

class SAD(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_filters,
        input_dim=80
    ):
        super(SAD, self).__init__()
        #import pdb;pdb.set_trace()
        self.inplanes = num_filters[0]
        self.conv1 = nn.Conv2d(1, num_filters[0] , kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(num_filters[0])


        self.layer1 = self._make_layer(block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=(2, 2))

        self.fc = nn.Linear(5120, 128)
        stats_input_dim = ((((input_dim + 1) // 2 + 1) // 2 + 1) // 2) * num_filters[-1]
        
        self.pooling = StatisticsPooling(input_dim=stats_input_dim)
        
        self.lstm_speaker_detection = nn.LSTM(
            128,
            128,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.lstm_combine = nn.LSTM(
            256,
            128,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.output_layer = nn.Linear(256, 1)
        #self.sigmoid = nn.Sigmoid()

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


    def forward(self, feats):
        # feats: (B, T, D)
        # target_speakers: (B, 4, 128)

        if len(feats.shape) == 3:
            feats = feats.unsqueeze(1)

        x = self.conv1(feats)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) # B, C, T, D

        x = x.transpose(-1, -2) # B, C, D, T
        x = x.reshape(x.shape[0], -1, x.shape[3]) # B, C*D, T
        #import pdb;pdb.set_trace()
        num_frames = x.shape[2]
        x = nn.functional.pad(x, (2, 2), mode="reflect")
        stats = []
        for i in range(num_frames):
            window = x[:, :, i:i+5]
            #print(window.shape, self.pooling.input_dim)
            pool = self.pooling(window)
            stats.append(pool)
        #print("stats:", stats[0].shape, len(stats))
        x = torch.stack(stats, dim=1) # B, T, C*D
        #print(x.shape)
        #import pdb;pdb.set_trace()
        x = self.fc(x)  # B, T, 128
        x = nn.functional.normalize(x, dim=2)

        b, t = x.shape[0], x.shape[1]

        sd_inputs = x # B, T, 128

        sd_outputs, _ = self.lstm_speaker_detection(sd_inputs) # B, T, 128
        #print(sd_outputs.shape)
        sd_outputs = sd_outputs.contiguous() # B, T, 128

        outputs, _ = self.lstm_combine(sd_outputs) # B, T, 256
        #print(outputs.shape)
        predictions = self.output_layer(outputs) #B, T, 4

        return predictions

def get_SAD():
    return SAD(
        SEBasicBlock,
        [3, 4, 6, 3],
        [32, 64, 128, 256],
    )
