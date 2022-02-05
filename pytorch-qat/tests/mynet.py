import torch       # ライブラリ「PyTorch」のtorchパッケージ
import torch.nn as nn  # 「ニューラルネットワーク」モジュールの別名定義

class QuantNet(nn.Module):
    def __init__(self, num_output_classes=10):
        super(QuantNet, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 4ch -> 8ch, 14x14 -> 7x7
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(8 * 7 * 7, 32)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(32, num_output_classes)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        # 1層目の畳み込み
        # 活性化関数 (activation) はReLU
        x = self.quant(x)
        x = self.conv1(x)
        x = self.relu1(x)

        # 縮小
        x = self.pool1(x)

        # 2層目+縮小
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # フォーマット変換 (Batch, Ch, Height, Width) -> (Batch, Ch)
        x = x.contiguous().view(x.shape[0], -1)

        # 全結合層
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.dequant(x)
        # x = F.log_softmax(x, dim=1)
        return x