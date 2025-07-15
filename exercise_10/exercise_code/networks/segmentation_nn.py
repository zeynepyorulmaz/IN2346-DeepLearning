"""SegmentationNN"""
import torch
import torch.nn as nn
import torchvision.models as models #Imported by me
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_mobilenet_v3_large
import torch.nn.functional as F


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x



class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23, hp=None):
        super().__init__()
        self.hp = hp
        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################


        # Load MobileNetV2 backbone
        mobilenet = models.mobilenet_v3_small(pretrained=True)
        self.encoder = mobilenet.features  # Output: (B, 1280, H/32, W/32)

        # Optional: reduce encoder output to 256 channels for decoder input
        self.bottleneck = nn.Sequential(
            nn.Conv2d(576, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )




        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 8x8 → 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #nn.Dropout(0.3),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # 16x16 → 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.Dropout(0.3),

            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),  # 32x32 → 64x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.Dropout(0.3),

            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),  # 64x64 → 128x128
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.Dropout(0.3),

            nn.ConvTranspose2d(64, num_classes, kernel_size=2, stride=2),  # 128x128 → 256x256
            nn.Upsample(size=(240, 240), mode='bilinear', align_corners=False)  # Final resize → 240x240
        )


        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        ########################################################################
        #                             YOUR CODE                                #  
        ########################################################################

        #x = self.model(x) # for current model
        # For backbone trained model
        #x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        #x = F.interpolate(x, size=(240, 240), mode='bilinear', align_corners=False)

        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return x

    # @property
    # def is_cuda(self):
    #     """
    #     Check if model parameters are allocated on the GPU.
    #     """
    #     return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(nn.Module):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1).long()  # Convert to int64/long
            # Clamp values to be within valid range [0, num_classes-1]
            y_tensor = torch.clamp(y_tensor, min=0, max=num_classes-1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=torch.float32)
            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image = target_image.clone().long()  # Ensure target is long type
        target_image[target_image == -1] = 0  # Map void label to background class

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()

if __name__ == "__main__":
    from torchinfo import summary
    summary(SegmentationNN(), (1, 3, 240, 240), device="cpu")