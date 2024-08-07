import tensorflow as tf
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import time

def generator_maker(data_name, file, batch_size, input_size, zero_weight=None):
    shuffle_indices = {}
    for radius in file.keys():
        radius_indices = {}
        for event_id in file[radius].keys():
            sub_ds = file[radius + "/" + event_id + "/labels"]
            radius_indices[int(event_id)] = tf.argsort(sub_ds, axis=0)
        shuffle_indices[int(radius)] = radius_indices
    def generator(events, radius):
        for event_no in events:
            groupname = str(radius) + "/" + str(event_no)
            sub_ds = file[groupname + "/" + data_name]
            if data_name == "quadruplets":
                sub_ds = tf.reshape(sub_ds[:,:,10:13], [sub_ds.shape[0], input_size]) / 1000
                sub_ds = tf.gather(sub_ds, shuffle_indices[radius][event_no])
                for i in range(0, sub_ds.shape[0]-batch_size, batch_size):
                    yield sub_ds[i:i+batch_size]
            elif data_name == "labels":
                if zero_weight is not None:
                    index1 = np.array(sub_ds)
                    sub_ds = np.empty(index1.shape)
                    sub_ds[index1.astype(bool)] = 1 - zero_weight
                    sub_ds[(1 - index1).astype(bool)] = zero_weight
                sub_ds = tf.gather(sub_ds, shuffle_indices[radius][event_no])
                for i in range(0, sub_ds.shape[0]-batch_size, batch_size):
                    yield sub_ds[i:min(i+batch_size, sub_ds.shape[0]-1)]
            else:
                raise Exception("Unrecognized data name")
    return generator

def get_zero_weight(file, radius):
    ntriplets = 0
    nmatches = 0
    for event_no in range(20):
        groupname = str(radius) + "/" + str(event_no)
        event_labels = file[groupname + "/labels"]
        nmatches += np.sum(event_labels)
        ntriplets += len(event_labels)
    return nmatches / ntriplets

def main(argv):
    # BEGIN INPUT
    
    inpath = "/pscratch/sd/m/max_zhao/policy/quadruplets_intersections.hdf5"
    metrics_path = "/global/homes/m/max_zhao/bin/policy_winter_metrics.csv"
    model_path = "/global/homes/m/max_zhao/mlkf/trackml/models/inter_policy/"
    checkpoints_path = "/global/homes/m/max_zhao/mlkf/trackml/models/checkpoints/inter_policy/"
    train_radius = 40
    test_radii = [40]
    test_threshold = 80
    train_event_thresholds = [80]
    batch_size = 4096
    nepochs = 50
    input_size = 15
    nevents = 100

    # END INPUT
    
    print("Devices:", tf.config.list_physical_devices('GPU'))

    infile = h5py.File(inpath, "r")

    zero_weight = get_zero_weight(infile, train_radius)
    print("Ratio of true to false labels:", zero_weight)
    class_weight = {0: zero_weight, 1: 1-zero_weight}

    triplet_gen = generator_maker("quadruplets", infile, batch_size, input_size)
    label_gen = generator_maker("labels", infile, batch_size, input_size)

    # for _ in range(10):
    #     gen = triplet_gen(range(80, 100), 40)
    #     for _ in range(30):
    #         print(next(gen))
    # assert False

    test_data_lst = []
    for test_radius in test_radii:
        test_args = [range(test_threshold, nevents), test_radius]
        test_triplets = tf.data.Dataset.from_generator(triplet_gen, args=test_args, output_signature=tf.TensorSpec(shape=[batch_size,input_size], dtype=tf.float32))
        test_labels = tf.data.Dataset.from_generator(label_gen, args=test_args, output_signature=tf.TensorSpec(shape=[batch_size], dtype=tf.int32))

        test_data = tf.data.Dataset.zip((test_triplets, test_labels))
        test_data = test_data.shuffle(buffer_size=1000)
        test_data_lst.append(test_data)

    training_data_lst = []
    for threshold in train_event_thresholds:
        train_args = [range(threshold), train_radius]
        train_triplets = tf.data.Dataset.from_generator(triplet_gen, args=train_args, output_signature=tf.TensorSpec(shape=[batch_size,input_size], dtype=tf.float32))
        train_labels = tf.data.Dataset.from_generator(label_gen, args=train_args, output_signature=tf.TensorSpec(shape=[batch_size], dtype=tf.int32))
        
        train_data = tf.data.Dataset.zip((train_triplets, train_labels))
        train_data = train_data.shuffle(buffer_size=1000)
        training_data_lst.append(train_data)
 
    #ds_iter = iter(training_data_lst[0])
    #for _ in range(5):
    #    print(next(ds_iter))

    for i, training_data in enumerate(training_data_lst):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=[input_size]),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])

        loss_fn = tf.keras.losses.BinaryCrossentropy()
        metric_names = [tf.keras.metrics.FalsePositives(), tf.keras.metrics.TruePositives(), tf.keras.metrics.FalseNegatives(), tf.keras.metrics.TrueNegatives()]
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss=loss_fn, metrics=metric_names)

        #history = model.fit(training_data, epochs=50, class_weight=class_weight, verbose=2, validation_data=test_data)
        
        metrics = []
        for epoch in range(nepochs):
            start_time = time.time()

            model.fit(training_data, epochs=1, class_weight=class_weight, verbose=2)
            model.save_weights(checkpoints_path + "epoch" + str(epoch))

            epoch_metrics = []
            for test_data in test_data_lst:
                eval_dict = model.evaluate(test_data, verbose=2)
                false_positive_rate = eval_dict[1] / (eval_dict[1] + eval_dict[4])
                false_negative_rate = eval_dict[3] / (eval_dict[3] + eval_dict[2])
                epoch_metrics += [false_positive_rate, false_negative_rate]
            
            metrics.append(epoch_metrics)
            # print(time.time() - start_time)
            print("Finished epoch {}".format(str(epoch+1)), epoch_metrics)
            print()
        
        model.save(model_path)
        #metrics = []
        #print(history.history.keys())
        #for key in history.history.keys():
        #    metrics.append(np.array(history.history[key]))
        metrics = np.array(metrics)
        np.savetxt(metrics_path, metrics, delimiter=",",)
        
        plt.figure(figsize=(8, 6))

        epochs = np.arange(1, metrics.shape[0]+1)
        plt.plot(epochs, metrics[:,0], label="False positive rate")
        plt.plot(epochs, metrics[:,1], label="False negative rate")

        plt.title("Training radius: 40 mm, Validation radius: 40 mm")
        plt.xlabel("Epoch")
        plt.ylabel("Rate (%)")

        plt.legend()
        plt.savefig("metrics.png")

        print("Finished training:", train_event_thresholds[i])

    infile.close()
    return 0

if __name__ == "__main__":
    print("\nFinished with exit code:", main(sys.argv))
