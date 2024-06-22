# -*- coding: utf-8 -*-

"""
実際に指パッチンを検出するファイル
以下で実行
python detection.py (実行する秒数)
"""

import pyaudio
import sys
import numpy as np
import tensorflow as tf
import os

project_dir = os.getcwd() + "/"
# 自作モジュールのimport
import detected_processing
sys.path.append(project_dir + "./my_modules/")
import learning
import const

"""
定数
"""
# 実行時に引数に時間を指定していたらその時間分実行し、指定していなかったらデフォルトで100秒
if len(sys.argv) == 2:
    RECORD_SECONDS = int(sys.argv[1]) 
else:
    RECORD_SECONDS = 100

"""
tensorflowに関する処理
"""
def set_tensorflow():
    # tensorflowで必要な変数を取得
    model, optimizer, loss_fn, accuracy_fn = learning.learning_algorithm(const.FOR_TENSORFLOW.DATA_LEN)
    
    # チェックポイントの初期化
    checkpoint = tf.train.Checkpoint(model=model)
    if const.FOR_PYAUDIO.RATE == 44100:
        checkpoint.restore(project_dir + "./model_data_44100/model.ckpt").expect_partial()
    elif const.FOR_PYAUDIO.RATE == 16000:
        checkpoint.restore(project_dir + "./model_data_16000/model.ckpt").expect_partial()

    return model

"""
pyaudioに関する処理
"""
def set_pyaudio():
    pa = pyaudio.PyAudio()
    
    stream = pa.open(
        format=const.FOR_PYAUDIO.FORMAT,
        channels=const.FOR_PYAUDIO.CHANNELS,
        rate=const.FOR_PYAUDIO.RATE,
        input=True,
        frames_per_buffer=const.FOR_PYAUDIO.chunk
    )

    return pa, stream

def main():
    model = set_tensorflow()
    pa, stream = set_pyaudio()

    # データを入れていく
    # allの長さは20以上にならない
    all = []
    
    # tmpは常に同じ長さ
    tmp = [False for k in range(0, 20)]
    
    print('指パッチンの検出を始めます')
    try:
        for i in range(0, int(const.FOR_PYAUDIO.RATE / const.FOR_PYAUDIO.chunk * RECORD_SECONDS)):
            data = stream.read(const.FOR_PYAUDIO.chunk)
            npData = np.frombuffer(data, dtype="int16") / 32768.0
        
            # npDataの中にthresoldより大きい数字があるかどうか
            threshold = 0.0002
            isThresholdOver = False
            print(max(npData))
            if max(npData) > threshold:
                isThresholdOver = True
        
            tmp.append(isThresholdOver)
            tmp.pop(0)
        
            # 9,10, 11がのどれかがtrueで他がfalseだけなら反応
            # iが11まではallの長さが足りないためエラーになる。
            if sum(tmp[9: 11]) >= 1 and sum(tmp) <= 3 and i >= 12:
        
                # 単発音の部分を取得する
                big_point_data = all[-10:-8] 
        
                big_point_data = np.frombuffer(b''.join(big_point_data), dtype="int16") / 32768.0
                X = np.fft.fft(big_point_data)
                amplitudeSpectrum = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in X]
        
                # 指パッチンである確率を算出
                result = model(np.array([amplitudeSpectrum]), training=False)
        
                print('これがフィンガースナップである確率>>' + str(result.numpy()[0][0]))
                if(result.numpy()[0] >= 0.5):
                    print('これは指パッチンです\n')
                else: 
                    print('これは指パッチンではないです\n')
        
                tmp = [False for k in range(0, 20)]
        
            all.append(data)
        
            # allに入れっぱなしだとメモリを食う気がしたので、20以上なら0番目を削除
            if len(all) >= 20:
                all.pop(0)
    except KeyboardInterrupt:
        print('指パッチン検出を終了します。')
    finally:
        stream.close()
        pa.terminate()

if __name__ == "__main__":
    main()