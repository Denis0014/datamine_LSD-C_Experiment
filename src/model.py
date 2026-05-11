"""
Модель полносвязной нейросети для классификации STL-10.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_simple_network(input_dim, num_classes, hidden_dims=None):
    """
    Создаёт простую полносвязную сеть с 1-2 скрытыми слоями.
    
    Args:
        input_dim: Размерность входа (для STL-10: 96*96*3 = 27648)
        num_classes: Количество классов (для STL-10: 10)
        hidden_dims: Список размеров скрытых слоёв (по умолчанию [512, 256])
    
    Returns:
        Скомпилированная модель Keras
    """
    if hidden_dims is None:
        hidden_dims = [512, 256]
    
    model = keras.Sequential()
    
    # Входной слой
    model.add(layers.InputLayer(shape=(input_dim,)))
    
    # Скрытые слои
    for hidden_dim in hidden_dims:
        model.add(layers.Dense(hidden_dim, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))
    
    # Выходной слой
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model


def create_network_for_iic(input_dim, num_classes):
    """
    Создаёт свёрточную сеть для IIC задачи.
    Выход: вероятности классов для оригинальных и аугментированных изображений.
    
    Args:
        input_dim: Размерность входа
        num_classes: Количество классов
    
    Returns:
        Модель, которая возвращает предсказания для оригинальных и аугментированных изображений
    """
    # Автоматически определяем размер (32 для CIFAR-10, 96 для STL-10)
    spatial_dim = int((input_dim / 3) ** 0.5) 
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Reshape((spatial_dim, spatial_dim, 3)),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling2D(),
        

        tf.keras.layers.Dense(256, activation='relu', name="embedding_layer"), 
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model


def create_network_for_lsd_c(input_dim, embedding_dim=256):
    """
    Создаёт сеть для LSD-C задачи.
    Выход: embedding для вычисления similitude между примерами.
    
    Args:
        input_dim: Размерность входа
        embedding_dim: Размерность эмбеддинга (по умолчанию 256)
    
    Returns:
        Модель, которая возвращает эмбеддинги
    """
    model = keras.Sequential([
        layers.InputLayer(shape=(input_dim,)),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(embedding_dim, activation=None),  # Без активации для эмбеддинга
    ])
    
    return model
    

def create_iic_model(input_dim, num_classes):
    """
    Создаёт модель для IIC задачи.
    
    Args:
        input_dim: Размерность входа
        num_classes: Количество классов
    
    Returns:
        Скомпилированная модель IICModel
    """
    backbone = create_network_for_iic(input_dim, num_classes)
    return backbone