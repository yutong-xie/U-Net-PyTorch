import torch
import torch.nn as nn
import torch.nn.functional as F


def Conv(Cin, Cout):
    layer = nn.Sequential(   
        nn.Conv2d(Cin, Cout , 3, padding = 1, stride = 1),
        nn.BatchNorm2d(Cout),
        nn.ReLU(inplace = True),
        nn.Conv2d(Cout, Cout, 3, padding = 1, stride = 1),
        nn.BatchNorm2d(Cout),
        nn.ReLU(inplace = True),
    )
    return layer

class MyModel(nn.Module):

    def __init__(self): 
        super(MyModel, self).__init__()
        self.down1 = Conv(3,64)
        self.pool = nn.MaxPool2d(2,2)
        self.down2 = Conv(64,128)
        self.down3 = Conv(128,256)
        self.down4 = Conv(256,512)
        self.down5 = Conv(512,1024)

        self.up1 =  nn.ConvTranspose2d(1024, 512, 3, stride = 2, padding =1, output_padding = 1)
        self.up2 =  nn.ConvTranspose2d(512, 256, 3, stride = 2, padding = 1, output_padding = 1)
        self.up3 =  nn.ConvTranspose2d(256, 128, 3, stride = 2, padding = 1, output_padding = 1)
        self.up4 =  nn.ConvTranspose2d(128, 64, 3, stride = 2, padding = 1, output_padding = 1)

        self.conv1 = Conv(1024, 512)
        self.conv2 = Conv(512, 256)
        self.conv3 = Conv(256, 128)
        self.conv4 = Conv(128, 64)
        self.conv5 = nn.Conv2d(64,9,1)

    def forward(self, x): 
        # convolution part 
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))      
        x3 = self.down3(self.pool(x2)) 
        x4 = self.down4(self.pool(x3))
        x5 = self.down5(self.pool(x4))

        
        # upsample and deconvolution part
        dx1 = self.up1(x5)
        tmp = torch.cat((x4,dx1), dim=1)
        dx1_1 = self.conv1(tmp)
        dx2 = self.up2(dx1_1)
        tmp = torch.cat((x3,dx2), dim=1)
        dx2_1 = self.conv2(tmp)

        dx3 = self.up3(dx2_1)
        tmp = torch.cat((x2,dx3), dim=1)
        dx3_1 = self.conv3(tmp)

        dx4 = self.up4(dx3_1)
        tmp = torch.cat((x1,dx4), dim=1)
        dx4_1 = self.conv4(tmp)

        out = self.conv5(dx4_1)
        
        return out 