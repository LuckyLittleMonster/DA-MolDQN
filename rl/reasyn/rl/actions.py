"""ReaSyn action generation for RL episodes."""

import torch


def get_reasyn_actions(smiles, models, fpindex, rxn_matrix, cfg_method,
                       top_k=20, return_scores=False):
    """Run FastSampler L1a on one molecule, return deduplicated candidate SMILES."""
    from ..chem.mol import Molecule
    from ..sampler.sampler_fast import FastSampler
    from ..utils.sample_utils import TimeLimit

    sampler_kwargs = {
        'use_fp16': cfg_method.use_fp16,
        'max_branch_states': cfg_method.max_branch_states,
        'skip_editflow': cfg_method.skip_editflow,
        'rxn_product_limit': cfg_method.rxn_product_limit,
    }
    evolve_kwargs = {
        'max_evolve_steps': cfg_method.max_evolve_steps,
        'num_cycles': cfg_method.num_cycles,
        'num_editflow_samples': cfg_method.num_editflow_samples,
        'num_editflow_steps': cfg_method.num_editflow_steps,
    }

    mol = Molecule(smiles)
    sampler = FastSampler(
        fpindex=fpindex, rxn_matrix=rxn_matrix,
        mol=mol, model=models,
        factor=cfg_method.search_width,
        max_active_states=cfg_method.expansion_width,
        **sampler_kwargs,
    )
    tl = TimeLimit(30)
    sampler.evolve(gpu_lock=None, time_limit=tl, **evolve_kwargs)
    torch.cuda.synchronize()

    df = sampler.get_dataframe()
    if len(df) == 0:
        return []

    df = df.drop_duplicates(subset='smiles')
    df = df.sort_values('score', ascending=False).head(top_k)
    if return_scores:
        return list(zip(df['smiles'].tolist(), df['score'].tolist(),
                        df['synthesis'].tolist())
                    )
    return df['smiles'].tolist()


def get_reasyn_actions_full(smiles, models, fpindex, rxn_matrix,
                            top_k=20):
    """Run Full ReaSyn (8x4 cycles + editflow, base Sampler) on one molecule.

    Used after training to get multi-step synthesis routes for final
    molecules.  Produces 2-7 step routes at ~60s/mol.  Requires fp32 models.
    """
    from ..chem.mol import Molecule
    from ..sampler.sampler import Sampler
    from ..utils.sample_utils import TimeLimit

    mol = Molecule(smiles)
    sampler = Sampler(
        fpindex=fpindex, rxn_matrix=rxn_matrix,
        mol=mol, model=models,
        factor=16,
        max_active_states=256,
    )
    tl = TimeLimit(120)
    sampler.evolve(
        gpu_lock=None, time_limit=tl,
        max_evolve_steps=8,
        num_cycles=4,
        num_editflow_samples=10,
        num_editflow_steps=100,
    )
    torch.cuda.synchronize()

    df = sampler.get_dataframe()
    if len(df) == 0:
        return []

    df = df.drop_duplicates(subset='smiles')
    df = df.sort_values('score', ascending=False).head(top_k)
    return list(zip(df['smiles'].tolist(), df['score'].tolist(),
                    df['synthesis'].tolist()))
