import torch
import torch.nn as nn

# import networks.all_models as all_models
import torchvision.models as models


class ResNetWithProjector(nn.Module):
    def __init__(self, args=None):
        super(ResNetWithProjector, self).__init__()
        self.n_classes = self.get_num_classes(args.loss, args.num_classes)
        self.args = args
        self.model = models.resnet34(weights="DEFAULT")
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_ftrs, self.n_classes)
        self.projector = nn.Sequential(
            nn.Linear(self.num_ftrs, self.num_ftrs),
            # nn.ReLU(inplace=True),
            nn.Linear(self.num_ftrs, 1024),
        )

    def forward(self, inputs, project=False):
        if not project:
            # Classification task
            x = self.model(inputs)
            return x, x
        else:
            x = self.model.conv1(inputs)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)

            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)
            features = self.projector(x)
            y = self.model.fc(x)

            return features, y

    def get_num_classes(self, loss, num_classes):
        """
        Get the number of classes based on the loss function.
        """
        class_modifiers = {
            "ce": 0,
            "binomial": 0,
            "ordinal_encoding": -1,
        }

        return num_classes + class_modifiers.get(loss, 0)


class ResNet(nn.Module):
    def __init__(self, args=None):
        super(ResNet, self).__init__()
        self.n_classes = self.get_num_classes(args.loss, args.num_classes)
        self.args = args
        self.model = models.resnet34(weights="DEFAULT")
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_ftrs, self.n_classes)

    def forward(self, inputs):
        x = self.model(inputs)
        return x, x

    def get_num_classes(self, loss, num_classes):
        """
        Get the number of classes based on the loss function.
        """
        class_modifiers = {
            "ce": 0,
            "binomial": 0,
            "ordinal_encoding": -1,
        }

        return num_classes + class_modifiers.get(loss, 0)
