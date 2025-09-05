import sys
import numpy as np

import pyscf
from pyscf import scf, gto

from wavefunction_analysis.utils import print_matrix
#from wavefunction_analysis.utils.pyscf_parser import read_molecule, build_atom

def get_spade(coeff_mo_in_ao, coeff_lo_in_ao, ovlp_ao, imp_lo_idx, nocc):
    s_half = get_ortho_basis(ovlp_ao)[0]
    coeff_lowdin = np.einsum('ij,jk->ik', s_half, mo_coeff)

    coeff_spade_imp, coeff_spade_env = [], []
    for (i0, i1) in [(0, nocc), (nocc, -1)]:
        coeff_lowdin_imp = coeff_lowdin[imp_lo_idx, i0:i1]
        u, s, vt = np.linalg.svd(coeff_lowdin_imp, full_matrices=True)
        print('u:', u.shape, 's:', s.shape, 'vt:', vt.shape)
        span = len(s)
        vt_span, vt_null = vt[:span], vt[span:]
        #c1 = np.einsum('ik,k,kj->ij', u, s, vt_span)
        #print('diff:', np.sum(c1-coeff_lowdin_imp))

        spade_imp = np.einsum('mj,kj->mk', coeff_lowdin_imp, vt_span)
        spade_env = np.einsum('mj,kj->mk', coeff_lowdin_imp, vt_null)
        coeff_spade_imp.append(spade_imp)
        coeff_spade_env.append(spade_env)

    return coeff_spade_imp, coeff_spade_env


def get_projection_diabatization(fock_in_ao, fock_in_lo, coeff_lo_in_ao,
                                 ovlp_ao, lo_idx, nelectrons, direction=1):
    if not isinstance(fock_in_lo, np.ndarray):
        s_half_inv = get_ortho_basis(ovlp_ao)[1]
        fock_in_lo = np.einsum('ij,jk,kl->il', s_half_inv, fock_in_ao, s_half_inv)

    imp_lo_idx, env_lo_idx = lo_idx
    faa = fock_in_lo[np.ix_(imp_lo_idx, imp_lo_idx)]
    fbb = fock_in_lo[np.ix_(env_lo_idx, env_lo_idx)]
    fab = fock_in_lo[np.ix_(imp_lo_idx, env_lo_idx)]

    es, vs = [], []
    for f in [faa, fbb]:
        e, v = np.linalg.eigh(f)
        es.append(e)
        vs.append(v)

    w = np.einsum('ji,jk,kl->il', vs[0], fab, vs[1])
    imp_canon = np.einsum('ij,mi->mj', vs[0], coeff_lo_in_ao[:,imp_lo_idx])
    env_canon = np.einsum('ij,mi->mj', vs[1], coeff_lo_in_ao[:,env_lo_idx])

    if direction == 1: # impurity occupied orbitals to environment virtual
        e = es[1][nelectrons[1]:, None] - es[0][:nelectrons[0]]
        #print_matrix('e0:', es[0])
        #print_matrix('e1:', es[1])
        #print_matrix('e:', e)
        amps = np.einsum('ia,ai->ai', w[:nelectrons[0], nelectrons[1]:], e)
        diff = np.einsum('ai,bi->ab', amps, amps)

        e, v = np.linalg.eigh(diff)
        env_canon[:,nelectrons[1]:] = np.einsum('ab,pa->pb', v, env_canon[:,nelectrons[1]:])

    elif direction == 2: # environment occupied orbitals to impurity virtual
        e = es[0][nelectrons[0]:, None] - es[1][:nelectrons[1]]
        amps = np.einsum('ai,ai->ai', w[nelectrons[0]:, :nelectrons[1]])
        diff = np.einsum('ai,aj->ij', amps, amps)

        e, v = np.linalg.eigh(diff)
        env_canon[:,:nelectrons[1]] = np.einsum('ij,pi->pj', v, env_canon[:,:nelectrons[1]])

    return [imp_canon, env_canon]


if __name__ == '__main__':
    atom = """
           O         0.4183272099    0.1671038379    0.1010361156
           H         0.8784893276   -0.0368266484    0.9330933285
           H        -0.3195928737    0.7774121014    0.3045311682
           O         3.0208058979    0.6163509592   -0.7203724735
           H         3.3050376617    1.4762564664   -1.0295977027
           H         2.0477791789    0.6319690134   -0.7090745711
           O         2.5143150551   -0.2441947452    1.8660305097
           H         2.8954132119   -1.0661605274    2.1741344071
           H         3.0247679096    0.0221180670    1.0833062723
    """

    frgm_idx = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    frag_basis_idx = np.array([0, 7, 14])

    basis = 'sto-3g'
    functional = 'pbe0'
    charge = 0

    mol = gto.Mole(
        atom = atom,
        basis = basis,
        verbose = 0,
        charge = charge,
        )

    mf = scf.RKS(mol)
    mf.max_cycle = 200
    mf.xc = functional
    e = mf.kernel()
    print('energy:', e)

    nocc = mol.nelectron // 2

    mo_coeff = mf.mo_coeff
    #print_matrix('mo_coeff:', mo_coeff)

    ovlp = mf.get_ovlp()
    #print_matrix('ovlp:', ovlp)

    s, v = np.linalg.eigh(ovlp)
    s_half = np.einsum('ik,k,jk->ij', v, s**.5, v)

    #s1 = np.einsum('ij,jk->ik', s_half, s_half)
    #print_matrix('s1:', s1)
    #print('diff:', np.sum(ovlp-s1))

    from wavefunction_analysis.entanglement.mol_lo_tools import partition_lo_to_imps
    from wavefunction_analysis.entanglement.fragment_entangle import get_localized_orbital, get_localized_orbital_rdm, get_embedding_orbital
    from wavefunction_analysis.utils import get_ortho_basis


    ovlp_ao = mf.get_ovlp()
    coeff_mo_in_ao = mf.mo_coeff

    extra_orb = 0

    # local orbital depends on the localization method
    coeff_lo_in_ao = get_localized_orbital(mol, coeff_mo_in_ao, method='lowdin')
    #print_matrix('coeff_lo_in_ao', coeff_lo_in_ao)
    #print('diff:', np.sum(coeff_lowdin-coeff_lo_in_ao))
    dm_lo_in_ao = get_localized_orbital_rdm(coeff_lo_in_ao, coeff_mo_in_ao, ovlp_ao, nocc, extra_orb=extra_orb)

    frgm_lo_idx = partition_lo_to_imps(frgm_idx, mol, coeff_lo_in_ao, min_weight=0.8)
    print('frgm_lo_idx:', frgm_lo_idx)

    ifrgm = 0
    embed_method = 0

    imp_lo_idx = frgm_lo_idx.copy()
    imp_lo_idx, env_lo_idx = np.array(imp_lo_idx.pop(ifrgm)), np.sort(np.concatenate(imp_lo_idx))
    neo_imp = len(imp_lo_idx)
    print('imp_lo_idx:', imp_lo_idx)
    print('env_lo_idx:', env_lo_idx)


    coeff_spade_imp, coeff_spade_env = get_spade(mo_coeff, None, ovlp, imp_lo_idx, nocc)
    print('coeff_spade_imp:', coeff_spade_imp[0].shape, coeff_spade_imp[1].shape, 'coeff_spade_env:', coeff_spade_env[0].shape, coeff_spade_env[1].shape)


    fock = mf.get_fock()
    imp_canon, env_canon = get_projection_diabatization(fock, None, coeff_lo_in_ao, ovlp, [imp_lo_idx, env_lo_idx], [5, 10])
    print('imp_canon:', imp_canon.shape, 'env_canon:', env_canon.shape)





    coeff_eo_in_ao, dm_eo_in_ao = get_embedding_orbital(dm_lo_in_ao, coeff_lo_in_ao,
                        ovlp_ao, imp_lo_idx, env_lo_idx, embed_method)

    #print_matrix('coeff_eo_in_ao', coeff_eo_in_ao)
    print('coeff_eo_in_ao:', coeff_eo_in_ao.shape)


    proj_imp = np.einsum('ik,il->kl', coeff_spade_imp, coeff_eo_in_ao)
