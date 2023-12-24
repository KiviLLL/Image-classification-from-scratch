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
• 下載附上的：從頭開始影像分類.ipynb (也可以不下載 按照以下步驟親自體驗)  
# 實作方法和說明  
• 將使用Keras 3在google colab進行以下操作：  
• 1.登入Google  
• 2.打開colab，輸入以下指令進行設定   
```
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
 ```
 !curl -O https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip
 !unzip -q kagglecatsanddogs_5340.zip
 !ls
 !ls PetImages
 ```  
   現在我們有一個PetImages包含兩個子資料夾的資料夾，Cat和Dog。每個子資料夾包含每個類別的圖像檔案  
• 4.過濾掉損壞的影像:  
    在處理大量現實世界影像資料時，損壞的影像是很常見的情況。讓我們過濾掉標題中不包含字串「JFIF」的編碼錯誤的圖像。  
    ```
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

• 5.訓練並評估模型的準確率:      
     訓練之前，使用KerasModel.compile配置和編譯模型。將optimizer類別設為adam，將loss設定為您先前定義的loss_fn函數，並透過將metrics參數設為 來accuracy指定要為模型評估的指標     
    使用Model.fit方法調整您的模型參數並最小化損失   準確度將近達到98% 
    想讓模型傳回機率，可以封裝經過訓練的模型，並將softmax 附加到該模型    
![img](https://github.com/KiviLLL/TensorFlow2.0/blob/main/img4.png)  
     最後恭喜！成功利用Keras API 借助預建資料集訓練了一個機器學習模型。     
# 參考資料
https://keras.io/examples/vision/image_classification_from_scratch/
