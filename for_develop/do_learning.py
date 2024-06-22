# -*- coding: utf-8 -*-

"""
学習を実行するファイル
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import os

project_dir = os.getcwd() + "/"
# 自作モジュールのimport
import sys
sys.path.append(project_dir + "my_modules/")
import learning
import sound_data
import const

np.random.seed(20160615)
tf.random.set_seed(20160615)

def set_tensorflow():
    # トレーニングデータを入れる変数の取得
    train_x = np.empty((0, const.FOR_TENSORFLOW.DATA_LEN), int)
    train_t = np.array([])
    
    # RATEによって取得するデータを変更
    if const.FOR_PYAUDIO.RATE == 44100:
        finger_wav_files = os.listdir(project_dir + "./sounds/finger-44100")
        not_finger_wav_files = os.listdir(project_dir + "./sounds/not-finger-44100")
        train_x, train_t = sound_data.add_train_data(project_dir 
                + "./sounds/finger-44100/", finger_wav_files, train_x, train_t, 1)
        train_x, train_t = sound_data.add_train_data(project_dir 
                + "./sounds/not-finger-44100/", not_finger_wav_files, train_x, train_t, 0)
    elif const.FOR_PYAUDIO.RATE == 16000:
        finger_wav_files = os.listdir(project_dir + "./sounds/finger-16000")
        not_finger_wav_files = os.listdir(project_dir + "./sounds/not-finger-16000")
        train_x, train_t = sound_data.add_train_data(project_dir 
                + "./sounds/finger-16000/", finger_wav_files, train_x, train_t, 1)
        train_x, train_t = sound_data.add_train_data(project_dir
                + "./sounds/not-finger-16000/", not_finger_wav_files, train_x, train_t, 0)
    
    train_t = train_t.reshape([len(train_t), 1])

    return train_x, train_t, finger_wav_files, not_finger_wav_files

"""
学習データの読み込み
"""
def get_tensorflow_data(model):
    checkpoint = tf.train.Checkpoint(model=model)
    if const.FOR_PYAUDIO.RATE == 44100:
        #checkpoint.restore(project_dir + "./model_data_44100/model.ckpt")
        print("44100")
    elif const.FOR_PYAUDIO.RATE == 16000:
        #checkpoint.restore(project_dir + "./model_data_16000/model.ckpt")
        print("16000")

    return checkpoint

"""
学習データの保存
"""
def save_tensorflow_data(checkpoint):
    if const.FOR_PYAUDIO.RATE == 44100:
        checkpoint.save(file_prefix=project_dir + "./model_data_44100/model.ckpt")
    elif const.FOR_PYAUDIO.RATE == 16000:
        checkpoint.save(file_prefix=project_dir + "./model_data_16000/model.ckpt")

    return checkpoint

def main():
    model, optimizer, loss_fn, accuracy_fn = learning.learning_algorithm(const.FOR_TENSORFLOW.DATA_LEN)
    train_x, train_t, finger_wav_files, not_finger_wav_files = set_tensorflow()
    checkpoint = get_tensorflow_data(model)

    # 学習の実行数
    learning_num = 4000
    
    # 学習の実行
    learning.execution(learning_num, model, optimizer, loss_fn, accuracy_fn, train_x, train_t)
    
    #print("正解データ数は:" + str(len(finger_wav_files)))
    #print("不正解データ数は:" + str(len(not_finger_wav_files)))
    #print("学習回数は: " + str(learning_num))

    save_tensorflow_data(checkpoint)
    
    # --- テスト ---
    #result = model(train_x[10:11], training=False)
    #print(type(train_x[0]))
    #result = model(train_x, training=False)
    #for i in range(len(result)):
    #    print(str(result[i].numpy()) + ":" + str(train_t[i]))
    
if __name__ == "__main__":
    main()