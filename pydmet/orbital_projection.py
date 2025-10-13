import sys
import numpy as np

import pyscf
from pyscf import scf, gto

from wavefunction_analysis.utils import print_matrix, get_ortho_basis
#from wavefunction_analysis.utils.pyscf_parser import read_molecule, build_atom

def get_spade(coeff_mo_in_ao, coeff_lo_in_ao, ovlp_ao, imp_lo_idx, nocc):
    coeff_lo_in_mo = np.einsum('mp,mn,nq->pq', coeff_lo_in_ao, ovlp_ao, coeff_mo_in_ao)

    vt_all = []
    coeff_spade_imp, coeff_spade_env = [], []
    # loop over for occupied and virtual orbitals
    for (i0, i1) in [(0, nocc), (nocc, coeff_lo_in_mo.shape[1])]:
        coeff_imp = coeff_lo_in_mo[imp_lo_idx, i0:i1]
        u, s, vt = np.linalg.svd(coeff_imp, full_matrices=True)
        #print('u:', u.shape, 's:', s.shape, 'vt:', vt.shape)
        print_matrix('spade singular values: '+str(np.sum(s**2)), s)
        span = len(s)
        vt_span, vt_null = vt[:span], vt[span:]
        #c1 = np.einsum('ik,k,kj->ij', u, s, vt_span)

        spade_imp = np.einsum('mj,kj->mk', coeff_imp, vt_span)
        spade_env = np.einsum('mj,kj->mk', coeff_imp, vt_null)
        coeff_spade_imp.append(spade_imp)
        coeff_spade_env.append(spade_env)

        vt_all.append(vt)
    return coeff_spade_imp, coeff_spade_env, vt_all


def get_projection_diabatization(fock_in_ao, coeff_lo_in_ao, ovlp_ao,
                                 lo_idx, nelectrons, direction=1, thresh=1e-6):
    """
    orbitals from projection-operator diabatization (POD)
    in the form: [[impurity occupied, impurity virtual], [environment occupied, environment virtual]]
    """
    fock_in_lo = np.einsum('mp,mn,nq->pq', coeff_lo_in_ao, fock_in_ao, coeff_lo_in_ao)

    imp_lo_idx, env_lo_idx = lo_idx
    aa, bb, ab = np.ix_(imp_lo_idx, imp_lo_idx), np.ix_(env_lo_idx, env_lo_idx), np.ix_(imp_lo_idx, env_lo_idx)
    faa, fbb, fab = fock_in_lo[aa], fock_in_lo[bb], fock_in_lo[ab]

    # block diagonalization
    energies, coeffs = [], []
    for f in [faa, fbb]:
        e, v = np.linalg.eigh(f)
        energies.append(e)
        coeffs.append(v)

    # upper right block coupling
    w = np.einsum('ji,jk,kl->il', coeffs[0], fab, coeffs[1])

    # transform fragement coefficients from LO to AO basis
    coeffs[0] = np.einsum('mp,pk->mk', coeff_lo_in_ao[:,imp_lo_idx], coeffs[0])
    coeffs[1] = np.einsum('mp,pk->mk', coeff_lo_in_ao[:,env_lo_idx], coeffs[1])

    if direction == 1: # impurity occupied orbitals to environment virtual
        e = energies[1][nelectrons[1]:, None] - energies[0][:nelectrons[0]]
        #print_matrix('e0:', es[0])
        #print_matrix('e1:', es[1])
        #print_matrix('e:', e)
        amps = np.einsum('ia,ai->ai', w[:nelectrons[0], nelectrons[1]:], e)
        diff = np.einsum('ai,bi->ab', amps, amps)

        e, v = np.linalg.eigh(diff)
        idx = np.where(np.abs(e)>thresh)[0][::-1] # order from large to small
        #print_matrix('e:', e[idx])
        v = np.einsum('ab,pa->pb', v[:,idx], coeffs[1][:,nelectrons[1]:])
        #coeffs[1] = np.concatenate((coeffs[1][:,:nelectrons[1]], v), axis=1)
        return ([coeffs[0][:,:nelectrons[0]], coeffs[0][:,nelectrons[0]:]], [coeffs[0][:,:nelectrons[1]], v])

    elif direction == 2: # environment occupied orbitals to impurity virtual
        e = es[0][nelectrons[0]:, None] - es[1][:nelectrons[1]]
        amps = np.einsum('ai,ai->ai', w[nelectrons[0]:, :nelectrons[1]], e)
        diff = np.einsum('ai,aj->ij', amps, amps)

        e, v = np.linalg.eigh(diff)
        idx = np.where(np.abs(e)>thresh)[0][::-1] # order from large to small
        v = np.einsum('ij,pi->pj', v[:,idx], coeffs[1][:,:nelectrons[1]])
        #coeffs[1] = np.concatenate((v, vectros[1][:,nelectrons[1]:]), axis=1)
        return ([coeffs[0][:,:nelectrons[0]], coeffs[0][:,nelectrons[0]:]], [v, coeffs[0][:,nelectrons[1]:]])


def get_solvent_contribution(mol, frgm_idx, coeff_eo_in_ao):
    weights = []

    aoslices = mol.aoslice_by_atom()
    for env_idx in frgm_idx:
        w = 0.
        for ia in env_idx:
            p0, p1 = aoslices[ia,2:]
            w += np.einsum('mi,mi->', coeff_eo_in_ao[p0:p1], coeff_eo_in_ao[p0:p1])
        weights.append(w)

    weights = np.array(weights)
    return (weights/np.linalg.norm(weights))



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
    geom = atom.split('\n')[1:]
    nelectrons = [5, 0]


    #atom = """
    #  C    0.0000000    0.5575780   -0.0238971
    #  C    0.0000000   -0.2378013    1.2726094
    #  H    0.9116742    1.1949582   -0.0524808
    #  H   -0.9116742    1.1949582   -0.0524808
    #  H    0.9050723   -0.8795601    1.3252643
    #  H   -0.9050723   -0.8795601    1.3252643
    #  H    0.0000000    0.4569013    2.1389522
    #  O    0.0000000   -0.3240002   -1.1112994
    #  H    0.0000000    0.2446943   -1.9245942
    #"""
    #frgm_idx = [[7,8], [0,1,2,3,4,5,6]]
    #nelectrons = [4, 0]

    basis = '3-21g'
    functional = 'pbe'
    charge = 0


    if len(sys.argv) > 1:
        from wavefunction_analysis.utils.sec_mole import read_symbols_coords
        from wavefunction_analysis.utils.pyscf_parser import build_atom

        xyzfile = sys.argv[1]
        symbols, coords = read_symbols_coords(xyzfile)
        atom = build_atom(symbols, coords)
        frgm_idx = [list(range(8))]
        for i in range(24):
            frgm_idx.append([8+i*3, 9+i*3, 10+i*3])

        geom = atom.split(';')[:-1]

        basis = 'def2-svpd'
        nelectrons[0] = 15


    mol = gto.M(
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
    nelectrons[1] = nocc - nelectrons[0]
    print('nelectrons:', nelectrons)

    from pydmet.dmet_tda import runtda
    nstates = 3
    imp_list = [frgm_idx[0], [x for l in frgm_idx[1:] for x in l]]

    atomimp = ''
    for i in imp_list[0]:
        atomimp += geom[i]
        atomimp += '\n'
    runtda(atom, atomimp, charge, imp_list, functional, basis, nstates, nelectrons=nelectrons)
    #sys.exit()


    from wavefunction_analysis.entanglement.mol_lo_tools import partition_lo_to_imps
    from wavefunction_analysis.entanglement.fragment_entangle import get_localized_orbital, get_localized_orbital_rdm, get_embedding_orbital
    from wavefunction_analysis.utils import get_ortho_basis


    ovlp_ao = mf.get_ovlp()
    coeff_mo_in_ao = mf.mo_coeff
    #print_matrix('coeff_mo_in_ao:', coeff_mo_in_ao)

    extra_orb = 0

    # local orbital depends on the localization method
    coeff_lo_in_ao = get_localized_orbital(mol, coeff_mo_in_ao, method='lowdin')
    #print_matrix('coeff_lo_in_ao', coeff_lo_in_ao)
    frgm_lo_idx = partition_lo_to_imps(frgm_idx, mol, coeff_lo_in_ao, min_weight=0.8)

    ifrgm = 0
    embed_method = 0

    imp_lo_idx = frgm_lo_idx.copy()
    imp_lo_idx, env_lo_idx = np.array(imp_lo_idx.pop(ifrgm)), np.sort(np.concatenate(imp_lo_idx))
    neo_imp = len(imp_lo_idx)
    print('imp_lo_idx:', imp_lo_idx.shape, '\n', imp_lo_idx)
    print('env_lo_idx:', env_lo_idx.shape, '\n', env_lo_idx)


    coeff_spade_imp, coeff_spade_env, vt_1 = get_spade(coeff_mo_in_ao, coeff_lo_in_ao, ovlp_ao, imp_lo_idx, nocc)
    print('coeff_spade_imp:', coeff_spade_imp[0].shape, coeff_spade_imp[1].shape, 'coeff_spade_env:', coeff_spade_env[0].shape, coeff_spade_env[1].shape)

    coeff_spade_imp, coeff_spade_env, vt_2 = get_spade(coeff_mo_in_ao, coeff_lo_in_ao, ovlp_ao, env_lo_idx, nocc)
    print('coeff_spade_imp:', coeff_spade_imp[0].shape, coeff_spade_imp[1].shape, 'coeff_spade_env:', coeff_spade_env[0].shape, coeff_spade_env[1].shape)

    #print_matrix('vt1:', vt_1[0].T)
    #print_matrix('vt2:', vt_2[0].T)
    #s2 = np.einsum('ik,jk->ij', vt_1[0], vt_2[0])
    #print_matrix('s2:', s2[:5,-5:])

    fock_in_ao = mf.get_fock()
    coeff_pod_imp, coeff_pod_env = get_projection_diabatization(fock_in_ao, coeff_lo_in_ao, ovlp_ao, [imp_lo_idx, env_lo_idx], nelectrons, direction=1)
    print('coeff_pod_imp:', coeff_pod_imp[0].shape, coeff_pod_imp[1].shape, 'coeff_pod_env:', coeff_pod_env[0].shape, coeff_pod_env[1].shape)

    weights = get_solvent_contribution(mol, frgm_idx, coeff_pod_env[1])
    print_matrix('weights:', weights)

    #print_matrix('ovlp:', np.einsum('mi,mj->ij', coeff_pod_imp[0], coeff_pod_env[1]))
