
import csv
import EnvironmentManager.EnvironmentManager as envMgr
import numpy as np
import os
import tensorflow as tf




if (__name__ == "__main__"):
    terminalState = False
    # configure TF and construct model
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # Set to -1 if CPU should be used CPU = -1 , GPU = 0

    gpus = tf.config.experimental.list_physical_devices('GPU')
    cpus = tf.config.experimental.list_physical_devices('CPU')
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(1, input_shape=[1,1]))
    model.compile()

    em = envMgr.EnvironmentManager()

    em.startEpisode()


    print("recDuration: ", em.snip.recDuration)

    with open('snip_train-none.tsv', 'r') as f:
        for line in csv.reader(f, delimiter='\t'):
            try:
                print(line[2])

            except Exception as inst:
                print(type(inst), ": Error reading from file.")
                f.close()
                break
            
         
