import tensorflow as tf
import sys
import h5py
import numpy as np

def generator_maker(data_name, file, batch_size, zero_weight=None):
    def generator(events, radius):
        for event_no in events:
            groupname = str(radius) + "/" + str(event_no)
            sub_ds = file[groupname + "/" + data_name]
            if data_name == "quadruplets":
                sub_ds = tf.reshape(sub_ds[:,1:,6:9], [sub_ds.shape[0], 9]) / 1000
                for i in range(0, sub_ds.shape[0]-batch_size, batch_size):
                    yield sub_ds[i:i+batch_size]
            #    for item in sub_ds:
            #        yield tf.reshape(item[:,6:9], [9]) / 1000
            elif data_name == "labels":
                if zero_weight is not None:
                    index1 = np.array(sub_ds)
                    sub_ds = np.empty(index1.shape)
                    sub_ds[index1.astype(bool)] = 1 - zero_weight
                    sub_ds[(1 - index1).astype(bool)] = zero_weight
                for i in range(0, sub_ds.shape[0]-batch_size, batch_size):
                    yield sub_ds[i:min(i+batch_size, sub_ds.shape[0]-1)]
            #    for item in sub_ds:
            #        yield item
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
    inpath = "/pscratch/sd/m/max_zhao/train_policy/quadruplets_comb.hdf5"
    metrics_path = "/global/homes/m/max_zhao/bin/metrics_80_30_50.csv"
    model_path = "/global/homes/m/max_zhao/mlkf/trackml/models/policy_80_30/"
    checkpoints_path = "/global/homes/m/max_zhao/mlkf/trackml/models/checkpoints/policy_80_30/"
    train_radius = 30
    test_radii = [50]
    train_event_thresholds = [80]
    batch_size = 1024
    
    print("Devices:", tf.config.list_physical_devices('GPU'))

    infile = h5py.File(inpath, "r")

    zero_weight = get_zero_weight(infile, train_radius)
    print("Ratio of true to false labels:", zero_weight)
    class_weight = {0: zero_weight, 1: 1-zero_weight}

    triplet_gen = generator_maker("quadruplets", infile, batch_size)
    label_gen = generator_maker("labels", infile, batch_size)
    

    test_data_lst = []
    for test_radius in test_radii:
        test_args = [range(80, 100), test_radius]
        test_triplets = tf.data.Dataset.from_generator(triplet_gen, args=test_args, output_signature=tf.TensorSpec(shape=[batch_size,9], dtype=tf.float32))
        test_labels = tf.data.Dataset.from_generator(label_gen, args=test_args, output_signature=tf.TensorSpec(shape=[batch_size], dtype=tf.int32))

        test_data = tf.data.Dataset.zip((test_triplets, test_labels))
        test_data = test_data.shuffle(buffer_size=1000)
        test_data_lst.append(test_data)

    training_data_lst = []
    for threshold in train_event_thresholds:
        train_args = [range(threshold), train_radius]
        train_triplets = tf.data.Dataset.from_generator(triplet_gen, args=train_args, output_signature=tf.TensorSpec(shape=[batch_size,9], dtype=tf.float32))
        train_labels = tf.data.Dataset.from_generator(label_gen, args=train_args, output_signature=tf.TensorSpec(shape=[batch_size], dtype=tf.int32))
        
        train_data = tf.data.Dataset.zip((train_triplets, train_labels))
        train_data = train_data.shuffle(buffer_size=1000)
        training_data_lst.append(train_data)
 
    #ds_iter = iter(training_data_lst[0])
    #for _ in range(5):
    #    print(next(ds_iter))

    for i, training_data in enumerate(training_data_lst):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=[9]),
            tf.keras.layers.Dense(72, activation="relu"),
            tf.keras.layers.Dense(72, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])

        loss_fn = tf.keras.losses.BinaryCrossentropy()
        metric_names = [tf.keras.metrics.FalsePositives(), tf.keras.metrics.TruePositives(), tf.keras.metrics.FalseNegatives(), tf.keras.metrics.TrueNegatives()]
        model.compile(optimizer="adam", loss=loss_fn, metrics=metric_names)

        #history = model.fit(training_data, epochs=50, class_weight=class_weight, verbose=2, validation_data=test_data)
        
        metrics = []
        for epoch in range(50):
            model.fit(training_data, epochs=1, class_weight=class_weight, verbose=2)
            model.save_weights(checkpoints_path + "epoch" + str(epoch))

            epoch_metrics = []
            for test_data in test_data_lst:
                eval_dict = model.evaluate(test_data, verbose=2)
                false_positive_rate = eval_dict[1] / (eval_dict[1] + eval_dict[4])
                false_negative_rate = eval_dict[3] / (eval_dict[3] + eval_dict[2])
                epoch_metrics += [false_positive_rate, false_negative_rate]
            
            metrics.append(epoch_metrics)
            print("Finished epoch:", epoch_metrics)
        
        model.save(model_path)
        #metrics = []
        #print(history.history.keys())
        #for key in history.history.keys():
        #    metrics.append(np.array(history.history[key]))
        np.savetxt(metrics_path, np.array(metrics), delimiter=",",)
        print("Finished training:", train_event_thresholds[i])

    infile.close()
    return 0

if __name__ == "__main__":
    print("\nFinished with exit code:", main(sys.argv))
