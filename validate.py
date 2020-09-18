
import csv
import EnvironmentManager.EnvironmentManager as envMgr
import numpy as np
import os
import tensorflow as tf

if (__name__ == "__main__"):
    terminalState = False

    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        # Disable first GPU
        tf.config.set_visible_devices(physical_devices[0:], 'GPU')
        logical_devices = tf.config.list_logical_devices('GPU')
        # Logical device was not created for first GPU
        assert len(logical_devices) == len(physical_devices) - 1
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    gpus = tf.config.experimental.list_physical_devices('GPU')
    cpus = tf.config.experimental.list_physical_devices('CPU')

    autoencoder = tf.keras.models.load_model('autoencoder-lstm.tfm')

    total_error = []

    with open('snip_test-eyem.tsv', 'r') as f:
        for line in csv.reader(f, delimiter='\t'):
            em = envMgr.EnvironmentManager()
            em.startEpisode(line[2])
            print(line[2], em.envId)
            dta = np.fromiter(em.snip, dtype=np.float)
            dta = np.resize(dta, int(len(dta)/250)*250)
            dta2 = dta.reshape((int(len(dta)/250),250,1))
            
            # for i in range(0,int(len(dta)/5)*5):
            #    for j in range(0, 5):
            #        dta2 = dta[(i*5)+j:(i+1)*5+j]
            #        dta2 = dta2.reshape((1,5,1))
            error = autoencoder.evaluate(dta2, dta2)

            print("MSE: ", error)         

            total_error.append(error)

    print(total_error)

