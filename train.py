"""
Реализация CIFAR-10 классификации с использованием LSD-C потери, Softmax, MSE и IIC.

Шаги:
1. Загружаем CIFAR-10 данные (или STL-10 как альтернатива)
2. Создаём простую полносвязную сеть (входной -> скрытые -> выходной слои)
3. Обучаем сеть на различных функциях ошибок (LSD-C, Softmax, MSE, IIC)
4. Анализируем результаты и сохраняем визуализацию
"""

import os
import sys
import argparse
import pickle

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from data_loader import load_cifar10_dataset, prepare_data_for_training
from model import create_network_for_lsd_c, create_simple_network
from losses import LSD_C_Loss, IIC_Loss, classification_accuracy_from_embeddings
from analyze import analyze_results, save_summary_report
from src.resnet18 import resnet18


def parse_args():
    parser = argparse.ArgumentParser(
        description="Обучение нейросети на CIFAR-10 с различными функциями потерь"
    )
    
    # Параметры данных
    parser.add_argument("--data-dir", type=str, default="data", help="Папка с данными CIFAR-10")
    
    # Параметры модели
    parser.add_argument("--embedding-dim", type=int, default=256, help="Размерность эмбеддинга")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate оптимизатора")
    
    # Параметры обучения
    parser.add_argument("--batch-size", type=int, default=32, help="Размер батча")
    parser.add_argument("--epochs", type=int, default=20, help="Количество эпох обучения")
    parser.add_argument("--validation-split", type=float, default=0.1, help="Доля валидации")
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2], help="Уровень логов")
    
    # Параметры сохранения
    parser.add_argument("--model-dir", type=str, default="results", help="Папка для сохранения модели")
    parser.add_argument("--results-dir", type=str, default="results", help="Папка для сохранения результатов")
    parser.add_argument("--model-name", type=str, default="cifar10_model", help="Имя файла модели")
    
    return parser.parse_args()


def save_model_and_history(model, history, model_dir, model_name):
    """Сохраняет модель и историю обучения."""
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}.keras")
    history_path = os.path.join(model_dir, f"{model_name}_history.pkl")
    
    model.save(model_path)
    print(f"✓ Модель сохранена в {model_path}")
    
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"✓ История обучения сохранена в {history_path}")
    
    return model_path


def main():
    args = parse_args()
    
    print("="*70)
    
    print("\nЗагрузка CIFAR-10 датасета")
    print("-"*70)
    
    (X_train, y_train), (X_test, y_test) = load_cifar10_dataset(args.data_dir)
    
    train_dataset, val_dataset, test_dataset, input_dim, num_classes = prepare_data_for_training(
        X_train, y_train, X_test, y_test, args.batch_size
    )
    
    # Cписок экспериментов
    experiments = [
        {'name': f"{args.model_name}_lsd_c", 'type': 'lsd_c'},
        {'name': f"{args.model_name}_softmax", 'type': 'softmax'},
        {'name': f"{args.model_name}_mse", 'type': 'mse'},
        # {'name': f"{args.model_name}_iic", 'type': 'iic'} # Показал плохие результаты, поэтому не участвует в основной работе
    ]

    results_root = args.results_dir
    last_trained_model = None
    last_history = None

    for exp in experiments:
        model_name = exp['name']
        exp_dir = os.path.join(results_root, model_name)
        os.makedirs(exp_dir, exist_ok=True)

        # Класс фиксации времени эпох
        class TimingCallback(keras.callbacks.Callback):
            def on_train_begin(self, logs=None):
                self.epoch_times = []
                self.train_start = None

            def on_epoch_begin(self, epoch, logs=None):
                if self.train_start is None:
                    self.train_start = __import__('time').perf_counter()
                self._epoch_start = __import__('time').perf_counter()

            def on_epoch_end(self, epoch, logs=None):
                t = __import__('time').perf_counter() - self._epoch_start
                self.epoch_times.append(t)

            def on_train_end(self, logs=None):
                self.train_time = (__import__('time').perf_counter() - self.train_start) if self.train_start is not None else sum(self.epoch_times)

        # --- ЭКСПЕРИМЕНТ 1: LSD-C ---
        if exp['type'] == 'lsd_c':
            print(f"\n[Эксперимент] {model_name}: LSD-C (эмбеддинги + kNN)")
            model = create_network_for_lsd_c(input_dim, args.embedding_dim)

            optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
            model.compile(optimizer=optimizer, loss=LSD_C_Loss(), metrics=[classification_accuracy_from_embeddings])

            early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7)

            timing = TimingCallback()
            history = model.fit(train_dataset, validation_data=val_dataset, epochs=args.epochs, callbacks=[early_stopping, reduce_lr, timing], verbose=args.verbose)

            embeddings_train = model.predict(X_train.reshape(X_train.shape[0], -1).astype(np.float32), verbose=0)
            embeddings_test = model.predict(X_test.reshape(X_test.shape[0], -1).astype(np.float32), verbose=0)

            from sklearn.neighbors import KNeighborsClassifier
            knn = KNeighborsClassifier(n_neighbors=15)
            knn.fit(embeddings_train, y_train.flatten())
            y_pred = knn.predict(embeddings_test)

            model_path = save_model_and_history(model, history, args.model_dir, model_name)
            last_trained_model, last_history = model, history

        # --- ЭКСПЕРИМЕНТ 2: SOFTMAX ---
        elif exp['type'] == 'softmax':
            print(f"\n[Эксперимент] {model_name}: Softmax-классификатор")
            clf = create_simple_network(input_dim, num_classes)
            optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
            clf.compile(optimizer=optimizer, loss=keras.losses.SparseCategoricalCrossentropy(), metrics=[keras.metrics.SparseCategoricalAccuracy()])

            early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7)

            timing = TimingCallback()
            history = clf.fit(train_dataset, validation_data=val_dataset, epochs=args.epochs, callbacks=[early_stopping, reduce_lr, timing], verbose=args.verbose)

            X_test_flat = X_test.reshape(X_test.shape[0], -1).astype(np.float32)
            y_probs = clf.predict(X_test_flat, verbose=0)
            y_pred = np.argmax(y_probs, axis=1)

            model_path = save_model_and_history(clf, history, args.model_dir, model_name)
            last_trained_model, last_history = clf, history

        # --- ЭКСПЕРИМЕНТ 3: MSE ---
        elif exp['type'] == 'mse':
            print(f"\n[Эксперимент] {model_name}: MSE (one-hot targets)")
            clf = create_simple_network(input_dim, num_classes)
            optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
            clf.compile(optimizer=optimizer, loss=keras.losses.MeanSquaredError(), metrics=[keras.metrics.CategoricalAccuracy()])

            early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7)

            train_ds_onehot = train_dataset.map(lambda x, y: (x, tf.one_hot(y, num_classes))).prefetch(tf.data.AUTOTUNE)
            val_ds_onehot = val_dataset.map(lambda x, y: (x, tf.one_hot(y, num_classes))).prefetch(tf.data.AUTOTUNE)

            timing = TimingCallback()
            history = clf.fit(train_ds_onehot, validation_data=val_ds_onehot, epochs=args.epochs, callbacks=[early_stopping, reduce_lr, timing], verbose=args.verbose)

            X_test_flat = X_test.reshape(X_test.shape[0], -1).astype(np.float32)
            y_probs = clf.predict(X_test_flat, verbose=0)
            y_pred = np.argmax(y_probs, axis=1)

            model_path = save_model_and_history(clf, history, args.model_dir, model_name)
            last_trained_model, last_history = clf, history

        # --- ЭКСПЕРИМЕНТ 4: IIC ---
        elif exp['type'] == 'iic':
            print(f"\n[Эксперимент] {model_name}: IIC (Максимизация взаимной информации + kNN)")
            
            model = resnet18(num_classes=num_classes)
            optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
            model.compile(optimizer=optimizer, loss=IIC_Loss(num_classes=num_classes))
            
            feature_extractor = keras.Model(
                inputs=model.input,
                outputs=model.layers[-2].output,
                name="feature_extractor"
            )
            
            def augment_image(img):
                img = tf.image.random_flip_left_right(img)
                img = tf.image.random_brightness(img, max_delta=0.2)
                img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
                img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
                if tf.random.uniform(()) > 0.5:
                    img = tf.image.rot90(img)
                noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=0.05)
                img = tf.clip_by_value(img + noise, 0.0, 1.0)
                return img
            
            def prepare_iic_dataset(images, batch_size, augment=True):
                ds_orig = tf.data.Dataset.from_tensor_slices(images.astype(np.float32))
                if augment:
                    ds_aug = ds_orig.map(lambda x: augment_image(x), num_parallel_calls=tf.data.AUTOTUNE)
                else:
                    ds_aug = ds_orig.map(lambda x: x, num_parallel_calls=tf.data.AUTOTUNE)
                
                def flatten(x):
                    return tf.reshape(x, (-1,)), 0
                
                ds_orig = ds_orig.map(flatten).batch(batch_size)
                ds_aug = ds_aug.map(flatten).batch(batch_size)
                ds = tf.data.Dataset.zip((ds_orig, ds_aug))
                ds = ds.map(lambda orig, aug: (
                    tf.concat([orig[0], aug[0]], axis=0),
                    tf.concat([orig[1], aug[1]], axis=0)
                ))
                ds = ds.prefetch(tf.data.AUTOTUNE)
                return ds
            
            split_idx = int(len(X_train) * (1 - args.validation_split))
            X_train_img = X_train[:split_idx]   # (None, 32, 32, 3)
            X_val_img = X_train[split_idx:]     # (None, 32, 32, 3)
            y_train_img = y_train[:split_idx]
            y_val_img = y_train[split_idx:]
            
            train_ds_iic = prepare_iic_dataset(X_train_img, args.batch_size, augment=True)
            val_ds_iic = prepare_iic_dataset(X_val_img, args.batch_size, augment=False)
            
            early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
            timing = TimingCallback()
            
            history = model.fit(
                train_ds_iic,
                validation_data=val_ds_iic,
                epochs=args.epochs,
                callbacks=[early_stopping, reduce_lr, timing],
                verbose=args.verbose
            )
            
            X_train_flat = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
            X_test_flat = X_test.reshape(X_test.shape[0], -1).astype(np.float32)
            
            embeddings_train = feature_extractor.predict(X_train_flat, verbose=0)
            embeddings_test = feature_extractor.predict(X_test_flat, verbose=0)
            
            from sklearn.neighbors import KNeighborsClassifier
            knn = KNeighborsClassifier(n_neighbors=15)
            knn.fit(embeddings_train, y_train.flatten())
            y_pred = knn.predict(embeddings_test)
            
            model_path = save_model_and_history(model, history, args.model_dir, model_name)
            last_trained_model, last_history = model, history

        else:
            continue

        train_time = getattr(timing, 'train_time', None)
        epoch_times = getattr(timing, 'epoch_times', None)
        analyze_results(history, y_test, y_pred, num_classes, results_dir=exp_dir)
        save_summary_report(history, y_test, y_pred, num_classes, model_path, results_dir=exp_dir, training_time=train_time, epoch_times=epoch_times, train_samples=X_train.shape[0])
        print(f"Результаты эксперимента '{model_name}' сохранены в {exp_dir}")
    
    print("\nОценка и анализ результатов последней модели")
    print("-"*70)
    
    if last_trained_model is not None:
        if last_trained_model.input_shape[-1] == 3072:  # полносвязная
            X_train_input = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
            X_test_input = X_test.reshape(X_test.shape[0], -1).astype(np.float32)
        else:  # свёрточная (ожидает (32,32,3))
            X_train_input = X_train.astype(np.float32)
            X_test_input = X_test.astype(np.float32)

        features_train = last_trained_model.predict(X_train_input, verbose=0)
        features_test = last_trained_model.predict(X_test_input, verbose=0)
        
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(features_train, y_train.flatten())
        y_pred = knn.predict(features_test)
        
        analyze_results(last_history, y_test, y_pred, num_classes, args.results_dir)
        
        final_model_path = os.path.join(args.model_dir, f"{args.model_name}.keras")
        save_summary_report(last_history, y_test, y_pred, num_classes, final_model_path, args.results_dir)
    
    print("\n" + "="*70)
    print("✓ ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ!")
    print("="*70)


if __name__ == '__main__':
    main()