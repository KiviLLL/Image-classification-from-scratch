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
• 2.打開colab，輸入以下指令進行設定   
```python
import os
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt
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
• 7.使用影像資料增強:      
   沒有大型圖像資料集時，最好的做法是透過對訓練圖像應用隨機但真實的變換（例如隨機水平翻轉或小型隨機旋轉）來人為地引入樣本多樣性。這有助於讓模型接觸訓練資料的不同方面，同時減緩過度擬合。  
```pyhon
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
• 8.預處理資料的兩種選擇(請選擇一種):  
   選項1：使其成為模型的一部分  
         使用此選項，您的資料增強將在設備上進行，與模型執行的其餘部分同步，這意味著它將受益於 GPU 加速。  
         請注意，資料增強在測試時處於非活動狀態，因此輸入樣本只會在 期間增強fit()，而不是在呼叫evaluate()或 時增強predict()。  
         如果您正在 GPU 上進行訓練，這可能是個不錯的選擇。   
   選項2：將其套用到資料集，以獲得產生批量增強影像的資料集  
         使用此選項，您的資料增強將在 CPU 上非同步發生，並在進入模型之前進行緩衝。  
          如果您正在 CPU 上進行訓練，這是更好的選擇，因為它使資料增強非同步且非阻塞  
         在我們的例子中，我們將選擇第二個選項。如果您不確定選擇哪一個，第二個選項（非同步預處理）始終是可靠的選擇。
```python
//選項一
inputs = keras.Input(shape=input_shape)
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)
...  # Rest of the model
```
```python
//選項二
augmented_train_ds = train_ds.map(
lambda x, y: (data_augmentation(x, training=True), y))
```

# 參考資料
https://keras.io/examples/vision/image_classification_from_scratch/
