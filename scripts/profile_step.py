"""Profile per-step timing breakdown for 256 mols."""
import time, sys, os
os.chdir('/shared/data1/Users/l1062811/git/DA-MolDQN')
sys.path.insert(0, '.')
from rdkit import Chem
from rdkit.Chem import QED
from rdkit import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
import src.cenv as cenv

# Load 256 mols
with open("Data/zinc_10000.txt") as f:
    smiles = [l.strip() for l in f][:256]
mols = [Chem.MolFromSmiles(s) for s in smiles]

# C++ env
env = cenv.Environment(["C", "O", "N"], [3, 5, 6], 3, 2048, cenv.Flags())
maintain = [-2] * 256

# Profile C++ batch (72 threads)
t0 = time.time()
vas, des = env.get_valid_actions_batch(mols, maintain, 72)
print(f"C++ batch 256 mols 72T: {time.time()-t0:.3f}s")

# Profile C++ batch (1 thread)
t0 = time.time()
vas, des = env.get_valid_actions_batch(mols, maintain, 1)
print(f"C++ batch 256 mols  1T: {time.time()-t0:.3f}s")

# Profile QED+SA serial
t0 = time.time()
for m in mols:
    QED.qed(m)
    sascorer.calculateScore(m)
print(f"QED+SA serial 256 mols: {time.time()-t0:.3f}s")

# Profile QED+SA with multiprocessing
from multiprocessing import Pool
def compute_qed_sa(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return 0.0, 10.0
    return QED.qed(m), sascorer.calculateScore(m)

t0 = time.time()
with Pool(16) as p:
    results = p.map(compute_qed_sa, smiles)
print(f"QED+SA Pool(16) 256 mols: {time.time()-t0:.3f}s")

t0 = time.time()
with Pool(32) as p:
    results = p.map(compute_qed_sa, smiles)
print(f"QED+SA Pool(32) 256 mols: {time.time()-t0:.3f}s")
