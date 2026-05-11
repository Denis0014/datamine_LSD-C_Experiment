"""
Анализ и визуализация результатов обучения.
"""

import os
# import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns


def analyze_results(history, y_test, y_pred, num_classes, results_dir='results'):
    """
    Анализирует результаты обучения и создаёт визуализацию.
    
    Args:
        history: История обучения (из model.fit)
        y_test: True labels
        y_pred: Predicted labels
        num_classes: Количество классов
        results_dir: Папка для сохранения результатов
    """
    os.makedirs(results_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("="*60)
    
    # Графики обучения
    plot_training_history(history, results_dir)
    
    # Метрики классификации
    print_classification_metrics(y_test, y_pred, num_classes)
    
    # Матрица ошибок
    plot_confusion_matrix(y_test, y_pred, num_classes, results_dir)
    
    # Статистика
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✓ Итоговая точность: {accuracy:.4f} ({accuracy*100:.2f}%)")


def plot_training_history(history, results_dir='results'):
    """Выводит графики истории обучения."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Тренировочные потери', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Валидационные потери', linewidth=2)
    axes[0].set_xlabel('Эпоха', fontsize=12)
    axes[0].set_ylabel('Потеря', fontsize=12)
    axes[0].set_title('Функция потерь', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    if 'accuracy' in history.history:
        axes[1].plot(history.history['accuracy'], label='Тренировочная точность', linewidth=2)
        axes[1].plot(history.history['val_accuracy'], label='Валидационная точность', linewidth=2)
        axes[1].set_xlabel('Эпоха', fontsize=12)
        axes[1].set_ylabel('Точность', fontsize=12)
        axes[1].set_title('Точность классификации', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(results_dir, 'training_history.png')
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    print(f"✓ График обучения сохранён: {filepath}")
    plt.close()


def plot_confusion_matrix(y_test, y_pred, num_classes, results_dir='results'):
    """Выводит матрицу ошибок."""
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=[str(i) for i in range(num_classes)],
                yticklabels=[str(i) for i in range(num_classes)])
    ax.set_xlabel('Предсказанный класс', fontsize=12)
    ax.set_ylabel('Истинный класс', fontsize=12)
    ax.set_title('Матрица ошибок', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    filepath = os.path.join(results_dir, 'confusion_matrix.png')
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    print(f"✓ Матрица ошибок сохранена: {filepath}")
    plt.close()


def print_classification_metrics(y_test, y_pred, num_classes):
    """Выводит детальные метрики классификации."""
    print("\n" + "-"*60)
    print("Подробные метрики классификации:")
    print("-"*60)
    
    class_names = [f"Класс {i}" for i in range(num_classes)]
    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    print(report)


def save_summary_report(history, y_test, y_pred, num_classes, model_path, results_dir='results', training_time=None, epoch_times=None, train_samples=None):
    """Сохраняет текстовый отчёт с результатами.

    Доп. параметры:
        training_time: общее время тренировки в секундах (float)
        epoch_times: список времен для каждой эпохи в секундах
        train_samples: количество обучающих образцов (int)
    """
    accuracy = accuracy_score(y_test, y_pred)
    
    with open(os.path.join(results_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("ОТЧЁТ О РЕЗУЛЬТАТАХ ОБУЧЕНИЯ\n")
        f.write("="*60 + "\n\n")
        
        f.write("ПАРАМЕТРЫ МОДЕЛИ:\n")
        f.write(f"  - Модель сохранена в: {model_path}\n")
        f.write(f"  - Количество классов: {num_classes}\n\n")
        
        f.write("РЕЗУЛЬТАТЫ:\n")
        f.write(f"  - Финальная точность: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"  - Финальная потеря: {history.history['loss'][-1]:.4f}\n")
        f.write(f"  - Валидационная потеря: {history.history['val_loss'][-1]:.4f}\n\n")
        
        if 'accuracy' in history.history:
            f.write(f"  - Финальная точность (тренировка): {history.history['accuracy'][-1]:.4f}\n")
            f.write(f"  - Валидационная точность: {history.history['val_accuracy'][-1]:.4f}\n\n")
        # Timing info
        if training_time is not None:
            epochs_run = len(epoch_times) if epoch_times is not None else len(history.history.get('loss', []))
            f.write("ТАЙМИНГ:\n")
            f.write(f"  - Общее время тренировки (сек): {training_time:.3f}\n")
            f.write(f"  - Эпох выполнено: {epochs_run}\n")
            if epoch_times is not None and len(epoch_times) > 0:
                avg_epoch = sum(epoch_times) / len(epoch_times)
                f.write(f"  - Среднее время эпохи (сек): {avg_epoch:.3f}\n")
                # поэпоховые времена
                f.write(f"  - Время по эпохам (сек): {', '.join([f'{t:.3f}' for t in epoch_times])}\n")
            if train_samples is not None and training_time > 0:
                throughput = (train_samples * epochs_run) / training_time
                f.write(f"  - Пропускная способность (samples/sec): {throughput:.2f}\n")
            f.write("\n")
        
        f.write("МЕТРИКИ ПО КЛАССАМ:\n")
        f.write(str(classification_report(y_test, y_pred, target_names=[f"Класс {i}" for i in range(num_classes)], digits=4)))
    
    print(f"✓ Отчёт сохранён: {os.path.join(results_dir, 'summary.txt')}")
