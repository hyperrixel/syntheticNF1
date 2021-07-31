"""
Project: Synthetic NF1 MRI Images (syn26010238)
Team: DCD (3430042)
Competition: Hack4Rare 2021
Description: this module contains FCGAN model pair
"""


import torch


class DCDDiscriminator(torch.nn.Module):
    """
    Discriminator model
    ===================
    """


    def __init__(self, ing_channels : int, depth_base : int, kernel_size : int,
                 flattened_size : int, use_bias : bool =False,
                 leaky_relu_slope : float =0.2):
        """
        Initialize an instance of the object
        ====================================

        Parameters
        ----------
        ing_channels : int
            Count of color channels of the input images.
        depth_base : int
            Base value of channel multiplication in convolutional layers.
        kernel_size : int
            Size of kernels to apply during convolutions.
        flattened_size : int
            Size of the fully connected layer. Depends on the input image's
            size and all other required parameters.
        use_bias : bool, optional (False if omitted)
            Whether or not to use bias nodes in the layers.
        leaky_relu_slope : float, optional (0.2 if omitted)
            Value to use as negative slope for LeakyReLU activation.
        """

        super().__init__()
        self.__leaky_relu_slope = leaky_relu_slope
        self.cnn1 = torch.nn.Conv2d(ing_channels, depth_base, kernel_size,
                                    stride=1, padding=0, bias=use_bias)
        self.cnn2 = torch.nn.Conv2d(depth_base, depth_base * 2, kernel_size,
                                    stride=1, padding=0, bias=use_bias)
        self.bn2 = torch.nn.BatchNorm2d(depth_base * 2)
        self.cnn3 = torch.nn.Conv2d(depth_base * 2, depth_base * 4, kernel_size,
                                    stride=1, padding=0, bias=use_bias)
        self.bn3 = torch.nn.BatchNorm2d(depth_base * 4)
        self.cnn4 = torch.nn.Conv2d(depth_base * 4, depth_base * 8, kernel_size,
                                    stride=1, padding=0, bias=use_bias)
        self.bn4 = torch.nn.BatchNorm2d(depth_base * 8)
        self.cnn5 = torch.nn.Conv2d(depth_base * 8, depth_base * 16,
                                    kernel_size, stride=1, padding=0,
                                    bias=use_bias)
        self.bn5 = torch.nn.BatchNorm2d(depth_base * 16)
        self.cnn6 = torch.nn.Conv2d(depth_base * 16, depth_base * 32,
                                    kernel_size, stride=2, padding=0,
                                    bias=use_bias)
        self.bn6 = torch.nn.BatchNorm2d(depth_base * 32)
        self.cnn7 = torch.nn.Conv2d(depth_base * 32, depth_base * 64,
                                    kernel_size, stride=2, padding=0,
                                    bias=use_bias)
        self.bn7 = torch.nn.BatchNorm2d(depth_base * 64)
        self.cnn8 = torch.nn.Conv2d(depth_base * 64, 1, kernel_size,
                                    stride=2, padding=0, bias=use_bias)
        self.fc1 = torch.nn.Linear(flattened_size, 1, bias=use_bias)


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass
        ====================

        Parameters
        ----------
        x : torch.Tensor
            Input data to be forwarded.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        x = torch.nn.functional.leaky_relu(self.cnn1(x),
                                           negative_slope=
                                           self.__leaky_relu_slope,
                                           inplace=True)
        x = torch.nn.functional.leaky_relu(self.bn2(self.cnn2(x)),
                                           negative_slope=
                                           self.__leaky_relu_slope,
                                           inplace=True)
        x = torch.nn.functional.leaky_relu(self.bn3(self.cnn3(x)),
                                           negative_slope=
                                           self.__leaky_relu_slope,
                                           inplace=True)
        x = torch.nn.functional.leaky_relu(self.bn4(self.cnn4(x)),
                                           negative_slope=
                                           self.__leaky_relu_slope,
                                           inplace=True)
        x = torch.nn.functional.leaky_relu(self.bn5(self.cnn5(x)),
                                           negative_slope=
                                           self.__leaky_relu_slope,
                                           inplace=True)
        x = torch.nn.functional.leaky_relu(self.bn6(self.cnn6(x)),
                                           negative_slope=
                                           self.__leaky_relu_slope,
                                           inplace=True)
        x = torch.nn.functional.leaky_relu(self.bn7(self.cnn7(x)),
                                           negative_slope=
                                           self.__leaky_relu_slope,
                                           inplace=True)
        x = torch.nn.functional.leaky_relu(self.cnn8(x),
                                           negative_slope=
                                           self.__leaky_relu_slope,
                                           inplace=True)
        x = torch.flatten(x, 1)
        return torch.sigmoid(self.fc1(x))


class DCDGenerator(torch.nn.Module):
    """
    Generator model
    ===============
    """


    def __init__(self, ing_channels : int, depth_base : int, kernel_size : int,
                 vector_dims : tuple, use_bias : bool = False,
                 leaky_relu_slope : float =0.2):
        """
        Initialize an instance of the object
        ====================================

        Parameters
        ----------
        ing_channels : int
            Count of color channels of the output images.
        depth_base : int
            Base value of channel multiplication in transposed convolutional
            layers.
        kernel_size : int
            Size of kernels to apply during transposed convolutions.
        vector_dims : tuple(int, int, int)
            Dimensions of the input vector.
        use_bias : bool, optional (False if omitted)
            Whether or not to use bias nodes in the layers.
        leaky_relu_slope : float, optional (0.2 if omitted)
            Value to use as negative slope for LeakyReLU activation.

        Raises
        ------
        AssertionError
            When the length of vector_dims is not 3.
        """

        super().__init__()
        assert len(vector_dims) == 3, 'Length vector_dims must be exactly 3.'
        self.__vector_dims = vector_dims
        self.__leaky_relu_slope = leaky_relu_slope
        self.fc1 = torch.nn.Linear(vector_dims[0],
                                  (vector_dims[0] * vector_dims[1]
                                  * vector_dims[2]), bias=use_bias)
        self.tcnn1 = torch.nn.ConvTranspose2d(vector_dims[0],
                                              depth_base * 64, kernel_size,
                                              stride=1, padding=0,
                                              bias=use_bias)
        self.bn1 = torch.nn.BatchNorm2d(depth_base * 64)
        self.tcnn2 = torch.nn.ConvTranspose2d(depth_base * 64, depth_base * 32,
                                              kernel_size, stride=1, padding=0,
                                              bias=use_bias)
        self.bn2 = torch.nn.BatchNorm2d(depth_base * 32)
        self.tcnn3 = torch.nn.ConvTranspose2d(depth_base * 32, depth_base * 16,
                                              kernel_size, stride=1, padding=0,
                                              bias=use_bias)
        self.bn3 = torch.nn.BatchNorm2d(depth_base * 16)
        self.tcnn4 = torch.nn.ConvTranspose2d(depth_base * 16, depth_base * 8,
                                              kernel_size, stride=1, padding=0,
                                              bias=use_bias)
        self.bn4 = torch.nn.BatchNorm2d(depth_base * 8)
        self.tcnn5 = torch.nn.ConvTranspose2d(depth_base * 8, depth_base * 4,
                                              kernel_size, stride=1, padding=0,
                                              bias=use_bias)
        self.bn5 = torch.nn.BatchNorm2d(depth_base * 4)
        self.tcnn6 = torch.nn.ConvTranspose2d(depth_base * 4, depth_base * 2,
                                              kernel_size, stride=2, padding=0,
                                              bias=use_bias)
        self.bn6 = torch.nn.BatchNorm2d(depth_base * 2)
        self.tcnn7 = torch.nn.ConvTranspose2d(depth_base * 2, depth_base,
                                              kernel_size, stride=2, padding=0,
                                              bias=use_bias)
        self.bn7 = torch.nn.BatchNorm2d(depth_base)
        self.tcnn8 = torch.nn.ConvTranspose2d(depth_base, ing_channels,
                                              kernel_size, stride=2, padding=2,
                                              bias=use_bias)


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass
        ====================

        Parameters
        ----------
        x : torch.Tensor
            Input data to be forwarded.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        x = self.fc1(x)
        x = x.view(-1, *self.__vector_dims)
        x = torch.nn.functional.leaky_relu(self.bn1(self.tcnn1(x)),
                                           negative_slope=
                                           self.__leaky_relu_slope,
                                           inplace=True)
        x = torch.nn.functional.leaky_relu(self.bn2(self.tcnn2(x)),
                                           negative_slope=
                                           self.__leaky_relu_slope,
                                           inplace=True)
        x = torch.nn.functional.leaky_relu(self.bn3(self.tcnn3(x)),
                                           negative_slope=
                                           self.__leaky_relu_slope,
                                           inplace=True)
        x = torch.nn.functional.leaky_relu(self.bn4(self.tcnn4(x)),
                                           negative_slope=
                                           self.__leaky_relu_slope,
                                           inplace=True)
        x = torch.nn.functional.leaky_relu(self.bn5(self.tcnn5(x)),
                                           negative_slope=
                                           self.__leaky_relu_slope,
                                           inplace=True)
        x = torch.nn.functional.leaky_relu(self.bn6(self.tcnn6(x)),
                                           negative_slope=
                                           self.__leaky_relu_slope,
                                           inplace=True)
        x = torch.nn.functional.leaky_relu(self.bn7(self.tcnn7(x)),
                                           negative_slope=
                                           self.__leaky_relu_slope,
                                           inplace=True)
        return torch.tanh(self.tcnn8(x))
