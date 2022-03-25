import torch.nn as nn

class LinearBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearBlock, self).__init__()
        self.linear_block = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.linear_block(x)
        return x

class FracClassifier(nn.Module):

    def __init__(
            self,
            encoder_channels,
            classifier_channels=(256, 128, 64),
            final_channels=2,
            linear_kernel=4096,
            p = 0.
    ):
        super(FracClassifier, self).__init__()
        
        # resnet gives 
        self.linear_kernel = linear_kernel
        self.initial_conv = nn.Conv3d(encoder_channels, 256, kernel_size=3,stride=1, padding=1)
        self.bn_init = nn.InstanceNorm3d(256, affine=True)
        self.drop_1 = nn.Dropout(p=p)
        
        self.initial_conv1 = nn.Conv3d(256, 128, kernel_size=3,stride=1, padding=1)
        self.bn_init1 = nn.InstanceNorm3d(128, affine=True)
        self.drop_2 = nn.Dropout(p=p)
        
        self.initial_conv2 = nn.Conv3d(128, 64, kernel_size=3,stride=1, padding=1)
        self.bn_init2 = nn.InstanceNorm3d(64, affine=True)
        self.drop_3 = nn.Dropout(p=p)
        
        self.initial_conv3 = nn.Conv3d(64, 8, kernel_size=3,stride=1, padding=1)
        self.bn_init3 = nn.InstanceNorm3d(8, affine=True)
        self.drop_4 = nn.Dropout(p=p)
        
        self.vector_shape = encoder_channels
        self.layer1 = LinearBlock(self.linear_kernel, classifier_channels[0])
        self.drop_5 = nn.Dropout(p=p)
        
        self.layer2 = LinearBlock(classifier_channels[0], classifier_channels[1])
        self.drop_6 = nn.Dropout(p=p)
        
        self.layer3 = LinearBlock(classifier_channels[1], classifier_channels[2])
        self.drop_7 = nn.Dropout(p=p)
        
        self.final_dense = nn.Linear(classifier_channels[2], final_channels)


    def forward(self, x):
        x = self.initial_conv(x)
        x = self.drop_1(self.bn_init(x))
        
        x = self.initial_conv1(x)
        x = self.drop_2(self.bn_init1(x))
        
        x = self.initial_conv2(x)
        x = self.drop_3(self.bn_init2(x))
        
        x = self.initial_conv3(x)
        x = self.drop_4(self.bn_init3(x))     

        x = x.view(x.shape[0], -1)

        x = self.drop_5(self.layer1(x))
        x = self.drop_6(self.layer2(x))
        x = self.drop_7(self.layer3(x))
        x = self.final_dense(x)

        return x