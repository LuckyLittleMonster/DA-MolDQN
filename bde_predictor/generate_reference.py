#!/usr/bin/env python
"""Generate TF reference predictions for PyTorch equivalence validation.
Run: conda run -n hpc311alf python bde_predictor/generate_reference.py
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
import numpy as np

def main():
    import tensorflow as tf
    import nfp
    from tensorflow.keras import layers
    from nfp.preprocessing.features import get_ring_size

    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    def atom_featurizer(atom):
        return str((atom.GetSymbol(), atom.GetNumRadicalElectrons(),
                     atom.GetFormalCharge(), atom.GetChiralTag(), atom.GetIsAromatic(),
                     get_ring_size(atom, max_size=6), atom.GetDegree(),
                     atom.GetTotalNumHs(includeNeighbors=True)))

    def bond_featurizer(bond, flipped=False):
        if not flipped:
            atoms = "{}-{}".format(bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol())
        else:
            atoms = "{}-{}".format(bond.GetEndAtom().GetSymbol(), bond.GetBeginAtom().GetSymbol())
        btype = str(bond.GetBondType())
        ring = 'R{}'.format(get_ring_size(bond, max_size=6)) if bond.IsInRing() else ''
        return " ".join([atoms, btype, ring]).strip()

    preprocessor = nfp.SmilesBondIndexPreprocessor(
        atom_features=atom_featurizer, bond_features=bond_featurizer)
    preprocessor.from_json(
        './BDE-db2/Example-BDE-prediction/model_3_tfrecords_multi_halo_cfc/preprocessor.json')

    class Slice(layers.Layer):
        def call(self, inputs):
            n = tf.shape(inputs)[1] // 2
            return inputs[:, :n, :]

    model = tf.keras.models.load_model(
        './BDE-db2/Example-BDE-prediction/model_3_multi_halo_cfc/best_model.hdf5',
        custom_objects={**nfp.custom_objects, 'Slice': Slice}, compile=False)

    test_smiles = [
        'c1ccc(O)cc1',
        'CC(=O)Oc1ccccc1O',
        'Oc1ccc(O)cc1',
        'CC(C)(C)c1cc(O)cc(C(C)(C)C)c1O',
        'O=C(O)c1ccccc1O',
    ]

    results = {}
    for smi in test_smiles:
        inp = preprocessor(smi)
        inp_data = {k: v.tolist() for k, v in inp.items()}

        def gen(s=smi):
            d = preprocessor(s)
            d['n_atom'] = len(d['atom'])
            d['n_bond'] = len(d['bond'])
            return d

        dataset = tf.data.Dataset.from_generator(
            lambda s=smi: iter([gen(s)]),
            output_signature={
                **preprocessor.output_signature,
                'n_atom': tf.TensorSpec(shape=(), dtype=tf.int32),
                'n_bond': tf.TensorSpec(shape=(), dtype=tf.int32)
            }).padded_batch(1, padding_values={
                **preprocessor.padding_values,
                'n_atom': tf.constant(0, dtype="int32"),
                'n_bond': tf.constant(0, dtype="int32")})

        pred = model.predict(dataset, verbose=False)
        results[smi] = {
            'inputs': inp_data,
            'predictions': pred[0].tolist(),
            'n_bonds': int(pred.shape[1]),
        }
        print(f"{smi}: n_bonds={pred.shape[1]}, BDE[0]={pred[0,0,0]:.4f}")

    os.makedirs('bde_predictor/tests', exist_ok=True)
    with open('bde_predictor/tests/reference_predictions.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved reference predictions for {len(results)} molecules")

if __name__ == '__main__':
    main()
