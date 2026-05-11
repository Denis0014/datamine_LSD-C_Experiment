import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential

class BasicBlock(layers.Layer):
    """Базовый Residual блок для ResNet-18 и ResNet-34."""
    def __init__(self, filter_num, stride=1, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        if stride != 1:
            self.downsample = Sequential([
                layers.Conv2D(filter_num, (1, 1), strides=stride)
            ])
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=False):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        output = layers.add([x, residual])
        output = tf.nn.relu(output)
        return output


class ResNet(Model):
    """Модель ResNet."""
    def __init__(self, layer_dims, num_classes=10, **kwargs):
        super().__init__(**kwargs)
        # Входной блок: conv 3x3, batch norm, relu, max pool
        self.stem = Sequential([
            layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
        ])

        self.layer1 = self._build_resblock(64,  layer_dims[0])
        self.layer2 = self._build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self._build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self._build_resblock(512, layer_dims[3], stride=2)

        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)

    def _build_resblock(self, filter_num, blocks, stride=1):
        """Строит группу из нескольких residual блоков."""
        res_blocks = Sequential()
        res_blocks.add(BasicBlock(filter_num, stride))
        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))
        return res_blocks

    def call(self, inputs, training=False, mask=None):
        x = self.stem(inputs)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        x = self.fc(x)
        return x


def resnet18(num_classes=10):
    """Фабричная функция для создания модели ResNet-18."""
    return ResNet(layer_dims=[2, 2, 2, 2], num_classes=num_classes)