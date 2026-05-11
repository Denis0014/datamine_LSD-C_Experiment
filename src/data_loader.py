"""
Загрузчик данных для STL-10 датасета.
"""

import os
import numpy as np
import tensorflow as tf
import keras.datasets


def load_cifar10_dataset(data_dir='data'):
    """
    Загружает CIFAR-10 датасет.
    
    CIFAR-10 содержит изображения размером 32x32 пикселей, 3 канала (RGB).
    Классы: самолёт, автомобиль, птица, кот, олень, собака, лягушка, лошадь, корабль, грузовик
    
    Args:
        data_dir: Путь к папке с данными
    
    Returns:
        (X_train, y_train), (X_test, y_test)
    """
    os.makedirs(data_dir, exist_ok=True)
    
    cache_file = os.path.join(data_dir, 'cifar10_cache.npz')
    
    if os.path.exists(cache_file):
        print(f"✓ Загружаю кэшированные данные из {cache_file}")
        data = np.load(cache_file)
        return (data['X_train'], data['y_train']), (data['X_test'], data['y_test'])
    
    print("Загружаю CIFAR-10 датасет...")
    
    try:
        (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
        y_train = y_train.flatten()
        y_test = y_test.flatten()
        
        print(f"Используется CIFAR-10 датасет")
        print(f"  - Размер X_train: {X_train.shape}")
        print(f"  - Размер X_test: {X_test.shape}")
        
    except Exception as e:
        print(f"Ошибка при загрузке: {e}")
        print("Генерирую случайные данные для демонстрации...")
        
        num_train = 5000
        num_test = 1000
        img_height, img_width = 32, 32
        num_classes = 10
        
        X_train = np.random.randint(0, 256, (num_train, img_height, img_width, 3), dtype=np.uint8)
        y_train = np.random.randint(0, num_classes, num_train)
        X_test = np.random.randint(0, 256, (num_test, img_height, img_width, 3), dtype=np.uint8)
        y_test = np.random.randint(0, num_classes, num_test)
    
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    
    np.savez(cache_file, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    print(f"✓ Данные кэшированы в {cache_file}")
    
    return (X_train, y_train), (X_test, y_test)


def prepare_data_for_training(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, batch_size=32):
    """
    Подготавливает данные для обучения (flatten для полносвязной сети).
    
    Args:
        X_train, y_train, X_test, y_test: Исходные данные
        batch_size: Размер батча
    
    Returns:
        train_dataset, val_dataset, test_dataset, input_shape, num_classes
    """
    original_shape = X_train.shape[1:]
    num_classes = len(np.unique(y_train))
    
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    X_train_flat = X_train_flat.astype(np.float32)
    X_test_flat = X_test_flat.astype(np.float32)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_flat, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(batch_size)
    
    val_split = int(0.1 * len(X_train_flat))
    val_X = X_train_flat[-val_split:]
    val_y = y_train[-val_split:]
    val_dataset = tf.data.Dataset.from_tensor_slices((val_X, val_y)).batch(batch_size)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test_flat, y_test)).batch(batch_size)
    
    input_dim = X_train_flat.shape[1]
    
    print(f"\n📊 Информация о данных:")
    print(f"  - Оригинальный shape: {original_shape}")
    print(f"  - Входной размер: {input_dim}")
    print(f"  - Количество классов: {num_classes}")
    print(f"  - Размер тренировочного набора: {len(X_train_flat)}")
    print(f"  - Размер тестового набора: {len(X_test_flat)}")
    print(f"  - Размер батча: {batch_size}")
    
    return train_dataset, val_dataset, test_dataset, input_dim, num_classes
