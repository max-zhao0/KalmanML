import tensorflow as tf
import numpy as np
import sys

def main(argv):
    print("Devices", tf.config.list_physical_devices())

    indir = "/global/homes/m/max_zhao/mlkf/trackml/data/triplets"
    training_prop = 0.8

    triplets = np.loadtxt(indir + "/triplets.csv", delimiter=",")
    labels = np.loadtxt(indir + "/labels.csv", delimiter=",")
    
    #triplets = triplets[:500000]
    #labels = labels[:500000]

    training_index = int(triplets.shape[0] * 0.8)
    triplet_points = triplets[:,6:9] / 3000

    training_triplets = tf.convert_to_tensor(triplet_points[:training_index])
    training_labels = tf.convert_to_tensor(labels[:training_index])
    valid_triplets = tf.convert_to_tensor(triplet_points[training_index:])
    valid_labels = tf.convert_to_tensor(labels[training_index:])
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    zero_weight = 0.5 * np.sum(labels) / labels.shape[0]
    print("Zero weight:", zero_weight)
    class_weights = {0: zero_weight,
                     1: 1 - zero_weight}

    loss_fn = tf.keras.losses.BinaryCrossentropy()

    model.compile(optimizer="adam", loss=loss_fn, metrics=[tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()])
    
    rates = []

    for _ in range(10):
        model.fit(training_triplets, training_labels, epochs=1, class_weight=class_weights, verbose=0)
        eval_dict = model.evaluate(valid_triplets, valid_labels, verbose=0)
        print(eval_dict)
        rates.append([eval_dict[1] / (labels.shape[0] - np.sum(labels)), eval_dict[2] / np.sum(labels)])

    np.savetxt("rates.csv", np.array(rates), delimiter=",")

    return 0

if __name__ == "__main__":
    print("\nFinished with exit code:", main(sys.argv)) 
