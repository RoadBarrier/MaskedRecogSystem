# 使用vgg16对人脸佩戴口罩图片分类

import tensorflow as tf
# import os
# import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16
# from tensorflow.keras.models import Sequential, Model

# 1.加载数据
data_dir = "data"
batch_size = 32
img_height = 64
img_width = 64

# 划分
# 训练dataset  img,label
# list(train_ds.take(1))  # ([32 img] [32 label]) => 1 batch
train_ds = tf.keras.utils.image_dataset_from_directory(data_dir,
                                                       validation_split=0.2,
                                                       subset="training",
                                                       seed=123,
                                                       image_size=(img_height,
                                                                   img_width),
                                                       batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(data_dir,
                                                     validation_split=0.2,
                                                     subset="validation",
                                                     seed=123,
                                                     image_size=(img_height,
                                                                 img_width),
                                                     batch_size=batch_size)

# type(train_ds)
class_names = train_ds.class_names
print(class_names)
num_classes = len(class_names)  # 几类 标签

# 配置缓存
AUTOTUNE = tf.data.AUTOTUNE
# tensorflow.python.data.ops.dataset_ops.PrefetchDataset
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 加载vgg16预训练模型 对 imagenet 1000
base_model = VGG16(include_top=False,
                   weights='imagenet',
                   input_shape=(img_height, img_width, 3))


# 在base_model后加上 自己项目的全连接
def add_new_last_layer(base_model, FC_SIZE, nb_classes):
    x = base_model.output  # 拿到卷基层之后的输出
    x = GlobalAveragePooling2D()(x)
    x = Dense(FC_SIZE, activation='relu')(x)  # 一层全连接层 FC_SIZE,这里查模型
    x = Dropout(0.5)(x)
    predictions = Dense(nb_classes, activation='softmax')(x)  # 根据自己需求设置分类
    model = Model(
        inputs=base_model.input,
        outputs=predictions)  # model跟预训练模型挂上关系,model和base_model是同一个内存地址
    return model


model = add_new_last_layer(base_model, 256, num_classes)
for layer in model.layers[:19]:
    layer.trainable = False  # 冻结

# 4.编译模型
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy'])

model.summary()

# 5.训练模型
epochs = 10
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# 6.打印模型准确度
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model.save('mask_VGG16_3.h5')
