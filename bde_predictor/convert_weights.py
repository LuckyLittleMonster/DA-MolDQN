#!/usr/bin/env python
"""Extract BDE-db2 TF model weights to NumPy .npz format.

Run once in hpc311alf env:
    conda run -n hpc311alf python bde_predictor/convert_weights.py
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np

def extract_weights(model_path, output_path):
    import tensorflow as tf
    import nfp
    from tensorflow.keras import layers

    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    class Slice(layers.Layer):
        def call(self, inputs):
            input_shape = tf.shape(inputs)
            num_bonds = input_shape[1] // 2
            return inputs[:, :num_bonds, :]

    custom_objects = {**nfp.custom_objects, 'Slice': Slice}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)

    weights = {}
    for var in model.variables:
        name = var.name.replace(':0', '')
        arr = var.numpy()
        weights[name] = arr
        print(f"{name}: {arr.shape} {arr.dtype}")

    np.savez(output_path, **weights)
    print(f"\nSaved {len(weights)} weights to {output_path}")

    # Verify
    loaded = np.load(output_path)
    assert len(loaded.files) == len(weights), f"Mismatch: {len(loaded.files)} vs {len(weights)}"
    total_params = sum(loaded[k].size for k in loaded.files)
    print(f"Verified: {len(loaded.files)} arrays, {total_params:,} total parameters")

if __name__ == '__main__':
    extract_weights(
        model_path='./BDE-db2/Example-BDE-prediction/model_3_multi_halo_cfc/best_model.hdf5',
        output_path='./bde_predictor/weights/bde_db2_model3.npz',
    )
