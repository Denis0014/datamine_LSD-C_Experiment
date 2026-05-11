"""
Реализация функции потерь LSD-C (Least Squares Discriminant Classification).
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import Loss

EPS = 1e-5

class IIC_Loss(Loss):
    """
    IIC loss with entropy regularization to prevent collapse.
    """
    def __init__(self, num_classes=10, temperature=0.5, entropy_weight=0.5, name='iic_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.temperature = temperature      # <1 увеличивает энтропию, >1 уменьшает
        self.entropy_weight = entropy_weight  # штраф за неравномерное распределение

    def call(self, y_true, y_pred):
        total_batch = tf.shape(y_pred)[0]
        batch_size = total_batch // 2
        y_pred = tf.cast(y_pred, tf.float32)

        z = y_pred[:batch_size]
        zt = y_pred[batch_size:]

        z = tf.nn.softmax(z / self.temperature, axis=1)
        zt = tf.nn.softmax(zt / self.temperature, axis=1)

        joint = tf.einsum('ij,ik->ijk', z, zt)
        joint = tf.reduce_sum(joint, axis=0)
        joint = (joint + tf.transpose(joint)) / 2.0
        joint = joint / (tf.reduce_sum(joint) + 1e-8)
        joint = tf.clip_by_value(joint, 1e-8, 1.0 - 1e-8)

        p_i = tf.reduce_sum(joint, axis=1, keepdims=True)
        p_j = tf.reduce_sum(joint, axis=0, keepdims=True)

        log_pi = tf.math.log(p_i)
        log_pj = tf.math.log(p_j)
        log_joint = tf.math.log(joint)
        mutual_info = tf.reduce_sum(joint * (log_pi + log_pj - log_joint))

        p_marginal = tf.squeeze(p_i)
        entropy = -tf.reduce_sum(p_marginal * tf.math.log(p_marginal + 1e-8))

        loss = -mutual_info - self.entropy_weight * entropy
        return loss

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'temperature': self.temperature,
            'entropy_weight': self.entropy_weight,
        })
        return config
    

def lsd_c_loss(y_true, y_pred):
    """
    LSD-C потеря для классификации.
    
    Внутри мини-батча устанавливаются «псевдо-связи» между парами,
    чьи признаки классов совпадают. Затем сеть учится предсказывать эти связи.
    
    Формула:
    L_LSD-C = -1/N * sum_{i,j} [ y_ij * log(p_ij) + (1 - y_ij) * log(1 - p_ij) ]
    
    где:
    - y_ij = 1 если класс i == класс j, иначе 0
    - p_ij = передсказанная вероятность сходства между примерами i и j (через сигмоиду от косинусного сходства эмбеддингов)
    
    Args:
        y_true: Истинные метки классов (batch_size,)
        y_pred: Предсказанные эмбеддинги (batch_size, embedding_dim)
    
    Returns:
        Скалярное значение потерь
    """
    batch_size = tf.shape(y_pred)[0]
    
    y_pred_normalized = tf.math.l2_normalize(y_pred, axis=1)
    
    similarity_matrix = tf.matmul(y_pred_normalized, y_pred_normalized, transpose_b=True)
    
    if len(y_true.shape) == 1:
        y_true_expanded = tf.expand_dims(y_true, 1)  # (batch_size, 1)
        y_true_matrix = tf.cast(tf.equal(y_true_expanded, tf.transpose(y_true_expanded)), tf.float32)
    else:
        y_true_matrix = tf.matmul(y_true, y_true, transpose_b=True)
        y_true_matrix = tf.cast(y_true_matrix > 0, tf.float32)
    
    probabilities = tf.sigmoid(similarity_matrix * 5)
    
    # Добавляем небольшой eps для избежания log(0)
    eps = 1e-7
    probabilities = tf.clip_by_value(probabilities, eps, 1 - eps)
    
    bce = -(y_true_matrix * tf.math.log(probabilities) + 
            (1 - y_true_matrix) * tf.math.log(1 - probabilities))
    
    mask = 1 - tf.eye(batch_size)
    bce = bce * mask
    
    num_pairs = tf.reduce_sum(mask)
    loss = tf.reduce_sum(bce) / (num_pairs + 1e-8)
    
    return loss


class LSD_C_Loss(keras.losses.Loss):
    """
    Пользовательский класс потерь для LSD-C.
    """
    def __init__(self, name="lsd_c_loss", **kwargs):
        super().__init__(name=name, **kwargs)
    
    def call(self, y_true, y_pred):
        return lsd_c_loss(y_true, y_pred)


def classification_accuracy_from_embeddings(y_true, y_pred):
    """
    Вычисляет точность классификации на основе эмбеддингов.
    Для каждого примера находим ближайший пример в батче с одинаковым классом.
    """
    batch_size = tf.shape(y_pred)[0]
    
    y_pred_normalized = tf.nn.l2_normalize(y_pred, axis=1)

    similarity_matrix = tf.matmul(y_pred_normalized, y_pred_normalized, transpose_b=True)
    
    similarity_matrix = similarity_matrix - tf.eye(batch_size) * 1e9
    
    most_similar_idx = tf.argmax(similarity_matrix, axis=1)
    
    y_true_pred = tf.gather(y_true, most_similar_idx)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_true, y_true_pred), tf.float32))
    
    return accuracy
