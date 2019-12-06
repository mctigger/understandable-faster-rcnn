from torch import nn as nn


class RCNN(nn.Module):
    """
    Refines the predictions of the RPN.
    """
    def __init__(self, in_channels, num_classes, in_size=(7, 7), num_channels=1024):
        """
        The RCNN can be adopted to the Faster-RCNN backbone through these 
        parameters.
        
        Args:
            nn ([type]): [description]
            in_channels ([type]): Number of channels of the input feature map
            num_classes ([type]): Number of target classes.
            in_size (tuple, optional): Size of the input feauture map after 
                RoI-Pooling. Defaults to (7, 7).
            num_channels (int, optional): Number of channels of the RCNN. 
                Defaults to 1024.
        """
        super(RCNN, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_channels*in_size[0]*in_size[1], num_channels),
            nn.ReLU(),
            nn.Linear(num_channels, num_channels),
            nn.ReLU(),
            nn.Linear(num_channels, 4 + (num_classes + 1))
        )

    def forward(self, rois):
        """
        Run RCNN on each given RoI in parallel.
        
        Args:
            rois (Tensor[P?, C, O, O]): The feature maps for the input RoIs.
        
        Returns:
            [Tuple[Tensor[P?, 4], Tensor[P?, N_cl]]]: The first tensor 
                represents the bbox offset from the RoI bbox to the real object
                bbox. The second tensor classifies the bbox into one of N_cl 
                classes with the first entry representing the background class.
        """

        # We collapse the width, height and channel dimension since we apply a 
        # fully connected layer now.
        x = rois.view(rois.size(0), -1)

        x = self.fc(x)
        reg = x[:, :4]
        cls = x[:, 4:]

        return reg, cls