
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as K
import cv2
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import deepsurvk
from sklearn.linear_model import LogisticRegression
import tkinter as tk
from tkinter import filedialog
import time
from tkinter import ttk
from tkinter import messagebox
import threading
def browse_path1():
    global path1
    path1 = filedialog.askopenfilename()
    print("Path to image 1:", path1)

def browse_path2():
    global path2
    path2 = filedialog.askopenfilename()
    print("Path to image 2:", path2)
def analyze_images():
    global path1, path2

    if path1 is None or path2 is None:
        messagebox.showerror("Error", "Please select both images.")
        return
    else:
        progress_bar.start()
        for i in range(101):
            progress_bar["value"] = i
            progress_bar.update()

        t = threading.Thread(target=analyze_images_thread)
        t.start()

def analyze_images_thread():

    # 导入图像
    img_1 = image.load_img(path1, target_size=(336, 336))
    img_1 = np.array(img_1)
    img_2 = image.load_img(path2, target_size=(336, 336))
    img_2 = np.array(img_2)
    img = np.hstack((img_1, img_2))
    img = img.reshape(1, 336, 672, 3)
    train_x_ = img.astype("float32") * 1 / 255
    train_x_ = train_x_[:, ::2, ::2]

    # 载入模型
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    model = keras.models.load_model('mobileV3-1.h5')
    input = train_x_
    last_conv_layer = model.get_layer('dense')
    iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
    model_out, last_conv_layer = iterate(input)
    outcomes = np.array(last_conv_layer)
    outcomes = pd.DataFrame(outcomes)

    conv_base = keras.models.load_model('mobileV3conv_base.h5')
    with tf.GradientTape() as tape:
        last_conv_layer = conv_base.get_layer('multiply_11')
        iterate = tf.keras.models.Model([conv_base.inputs], [conv_base.output, last_conv_layer.output])
        model_out, last_conv_layer = iterate(input)
        grads = tape.gradient(model_out, last_conv_layer)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = 3 * heatmap
    heatmap[heatmap > 1] = 1
    heatmap = heatmap.reshape((11, 21))

    INTENSITY = 0.5

    raw = train_x_ * 255
    # raw = raw.resize(168, 336)
    heatmap = cv2.resize(heatmap, (336, 168))
    if model.predict(input) > 0.5:
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_WINTER)
    else:
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    img_combine = ((heatmap + np.array(raw)) * INTENSITY) / 255

    # 比例风险
    X_train = np.load('X_train.npy')
    Y_train = np.load('Y_train.npy')
    E_train = np.load('E_train.npy')
    X_train = pd.DataFrame(X_train)
    Y_train = pd.DataFrame(Y_train)
    E_train = pd.DataFrame(E_train)
    n_patients_train = X_train.shape[0]
    n_features = X_train.shape[1]

    cols_standardize = X_train.iloc[:, 0:256].columns
    X_ct = ColumnTransformer([('standardizer', StandardScaler(), cols_standardize)])
    X_ct.fit(X_train[cols_standardize])
    X_train[cols_standardize] = X_ct.transform(X_train[cols_standardize])
    outcomes[cols_standardize] = X_ct.transform(outcomes[cols_standardize])

    sort_idx = np.argsort(Y_train.to_numpy(), axis=None)[::-1]
    X_train = X_train.iloc[sort_idx, :]
    Y_train = Y_train.iloc[sort_idx, :]
    E_train = E_train.iloc[sort_idx, :]

    E_train1 = pd.DataFrame(E_train.iloc[:, 0])
    E_train2 = pd.DataFrame(E_train.iloc[:, 1])

    params = {'n_layers': 2,
              'n_nodes': 8,
              'activation': 'selu',
              'learning_rate': 0.154,
              'decays': 5.667e-3,
              'momentum': 0.887,
              'dropout': 0.5,
              'optimizer': 'nadam'}
    dsk = deepsurvk.DeepSurvK(n_features=n_features,
                              E=E_train1,
                              **params)
    loss = deepsurvk.negative_log_likelihood(E_train1)
    dsk.compile(loss=loss)
    callbacks = deepsurvk.common_callbacks()

    epochs = 1000
    history = dsk.fit(X_train, Y_train,
                      batch_size=n_patients_train,
                      epochs=epochs,
                      shuffle=False)
    Y_pred_train = np.exp(-dsk.predict(X_train))
    clf = LogisticRegression()
    clf.fit(Y_pred_train, E_train2)

    Y_pred_test = np.exp(-dsk.predict(outcomes))
    Y_pred_test = clf.predict_proba(Y_pred_test.reshape(-1, 1))[:, 0]
    Y_pred_test = round(Y_pred_test[0], 3)

    img_combine = img_combine.reshape(168, 336, 3)

    words2 = "Natural conception difficulty [NCD] Probability: " + str(Y_pred_test)
    words3 = "*NCD<0.57: Low risk; > 0.57: High risk"
    img_combine = cv2.resize(img_combine, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    if Y_pred_test < 0.57:
        words = "Low risk"
        cv2.putText(img_combine, words, (10, 250), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        words = "High risk"
        cv2.putText(img_combine, words, (10, 250), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.putText(img_combine, words2, (10, 280), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img_combine, words3, (10, 300), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("predict result", img_combine)
    progress_bar.stop()
    cv2.waitKey()




# 创建一个主窗口
root = tk.Tk()
root.title("Hysteroscopic Image Analysis")
root.geometry("800x600")

# 创建一个框架用于放置按钮和标签
frame = tk.Frame(root)
frame.pack()

# 创建一个标签
label = tk.Label(frame, text="Please select the images to analyze:", font=("Helvetica", 20))
label.pack()

# 创建两个按钮，用于选择图片
button1 = tk.Button(frame, text="Enter the image of the uterine cavity", font=("Helvetica", 20, "bold"), command=browse_path1)
button1.pack(pady=10)

button2 = tk.Button(frame, text="Enter the image of the uterine corner", font=("Helvetica", 20, "bold"), command=browse_path2)
button2.pack(pady=10)

# 创建一个按钮，用于分析所选的两张图片
analyze_button = tk.Button(frame, text="Analyze Images", font=("Helvetica", 20, "bold"), command=analyze_images)
analyze_button.pack(pady=20)

progress_bar = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=400, mode='determinate')
progress_bar.pack(pady=20)

# 运行主循环
root.mainloop()






