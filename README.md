# 從頭開始影像分類
• 南華大學-人工智慧期末報告  
• 11024156林昀皞  
# 目錄
•大綱  
•準備資源  
•實作方法和說明  
# 大綱
• 在 Kaggle Cats vs Dogs 資料集上從頭開始訓練影像分類器    
• 此範例展示如何從頭開始進行影像分類，從磁碟上的 JPEG 影像檔案開始，而不利用預先訓練的權重或預製的 Keras 應用程式模型。我們在 Kaggle Cats vs Dogs 二元分類資料集上示範了工作流程    
• 我們使用此image_dataset_from_directory實用程式產生資料集，並使用 Keras 影像預處理層進行影像標準化和資料增強。  
# 準備資源
• 請準備一個可以使用google colab的帳號   
# 實作方法和說明  
• 將使用Keras 3在google colab進行以下操作：  
• 1.登入Google  
• 2.打開colab，輸入以下指令進行設定keras 3   
```python
import os
import numpy as np
import keras
import tensorflow as tf
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt
```
• 3.載入資料：貓狗大戰資料集  
   原始資料下載並解壓縮  
 ```python
 !curl -O https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip
 !unzip -q kagglecatsanddogs_5340.zip
 !ls
 !ls PetImages
 ```
![img](https://github.com/KiviLLL/Image-classification-from-scratch/blob/KiviLLL-patch-1/img1.png)  
   現在我們有一個PetImages包含兩個子資料夾的資料夾，Cat和Dog。每個子資料夾包含每個類別的圖像檔案  
• 4.過濾掉損壞的影像:  
   在處理大量現實世界影像資料時，損壞的影像是很常見的情況。讓我們過濾掉標題中不包含字串「JFIF」的編碼錯誤的圖像。  
 ```python
num_skipped = 0
for folder_name in ("Cat", "Dog"):
folder_path = os.path.join("PetImages", folder_name)
for fname in os.listdir(folder_path):
fpath = os.path.join(folder_path, fname)
 try:
 fobj = open(fpath, "rb")
is_jfif = b"JFIF" in fobj.peek(10)
finally:
fobj.close()
if not is_jfif:
num_skipped += 1
# Delete corrupted image
os.remove(fpath)
print(f"Deleted {num_skipped} images.")
 ```
![img](https://github.com/KiviLLL/Image-classification-from-scratch/blob/KiviLLL-patch-1/img2.png)  

• 5.生成一個Dataset:      
```python
image_size = (180, 180)
batch_size = 128

train_ds, val_ds = keras.utils.image_dataset_from_directory(
    "PetImages",
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
```
![img](https://github.com/KiviLLL/Image-classification-from-scratch/blob/KiviLLL-patch-1/img4.png)  
• 6.視覺化數據:      
   以下是訓練資料集中的前 9 張圖像。         
```python
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(images[i]).astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
```  
![img](https://github.com/KiviLLL/Image-classification-from-scratch/blob/KiviLLL-patch-1/%E4%B8%8B%E8%BC%89%20(1).png)   
• 7.使用影像資料增強:      
   沒有大型圖像資料集時，最好的做法是透過對訓練圖像應用隨機但真實的變換（例如隨機水平翻轉或小型隨機旋轉）來人為地引入樣本多樣性。這有助於讓模型接觸訓練資料的不同方面，同時減緩過度擬合。  
```python
data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]


def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images
```  
  data_augmentation 讓我們透過重複應用資料集中的前幾張影像來視覺化增強樣本的樣子：  
```python
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(augmented_images[0]).astype("uint8"))
        plt.axis("off")
```
![img](https://github.com/KiviLLL/Image-classification-from-scratch/blob/KiviLLL-patch-1/%E4%B8%8B%E8%BC%89%20(2).png) 
• 8.處理資料:       
   將其套用到資料集，以獲得產生批量增強影像的資料集  
   使用此選項，您的資料增強將在 CPU 上非同步發生，並在進入模型之前進行緩衝。  
   如果您正在 CPU 上進行訓練，這是更好的選擇，因為它使資料增強非同步且非阻塞  
```python
augmented_train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x), y))
```
• 9.配置資料集以提高效能:       
   將資料增強應用於我們的訓練資料集，並確保使用緩衝預取，以便我們可以從磁碟生成數據，而不會導致 I/O 阻塞：  
```python
# Apply `data_augmentation` to the training images.
train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf_data.AUTOTUNE,
)
# Prefetching samples in GPU memory helps maximize GPU utilization.
train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
val_ds = val_ds.prefetch(tf_data.AUTOTUNE)
```
• 10.建立模型:       
   我們將建立一個小型版本的 Xception 網路。  
   注意：
   我們從預處理器開始模型data_augmentation，然後是一層 Rescaling。  
   我們Dropout在最終分類層之前添加了一個層。  
```python
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        units = 1
    else:
        units = num_classes

    x = layers.Dropout(0.25)(x)
    # We specify activation=None so as to return logits
    outputs = layers.Dense(units, activation=None)(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,), num_classes=2)
keras.utils.plot_model(model, show_shapes=True)
```
https://github.com/KiviLLL/Image-classification-from-scratch/blob/KiviLLL-patch-1/%E4%B8%8B%E8%BC%89%20(3).png
• 11.訓練AI模型:       
```python
epochs = 10

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
]
model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy(name="acc")],
)
model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)
```
![img](https://github.com/KiviLLL/Image-classification-from-scratch/blob/KiviLLL-patch-1/img5.png)   
   理論上完整資料集上訓練 25 個 epoch 後驗證準確率達到了 >90%（但google colab免費RAM跑不到25個，建議訓練在10個內）。  
• 12.對新數據進行推理:       
   請注意，資料增強和遺失在推理時處於非活動狀態。  
```python
img = keras.utils.load_img("PetImages/Cat/6779.jpg", target_size=image_size)
plt.imshow(img)

img_array = keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = float(keras.activations.sigmoid(predictions[0][0]))
print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")
```
![img](https://github.com/KiviLLL/Image-classification-from-scratch/blob/KiviLLL-patch-1/img6.png)   
# 參考資料
https://keras.io/examples/vision/image_classification_from_scratch/
