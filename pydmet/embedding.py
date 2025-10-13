import numpy as np
from wavefunction_analysis.utils import print_matrix

class EmbeddingProblem(object):
    pass


class EmbeddingBasis(object):
    pass


def emb_basis_dmet(coeff_lo_in_ao, dm_lo, lo_idx, thresh=1e-12):
    """
    # everything is in localized orbital (LO) basis
    """
    imp_lo_idx, env_lo_idx = lo_idx
    nlo_imp, nlo_env = len(imp_lo_idx), len(env_lo_idx)

    imp_imp_lo_ix = np.ix_(imp_lo_idx, imp_lo_idx)
    env_env_lo_ix = np.ix_(env_lo_idx, env_lo_idx)
    imp_env_lo_ix = np.ix_(imp_lo_idx, env_lo_idx)

    dm_imp_imp_lo = dm_lo[imp_imp_lo_ix]
    dm_env_env_lo = dm_lo[env_env_lo_ix]
    dm_imp_env_lo = dm_lo[imp_env_lo_ix]

    u, s, vt = np.linalg.svd(dm_imp_env_lo, full_matrices=True)
    arg = np.where(s>thresh)[0]

    coeff_imp = coeff_lo_in_ao[:, imp_lo_idx]
    coeff_env = np.dot(coeff_lo_in_ao[:, env_lo_idx], vt[arg].T)
    coeff_eo_in_ao = np.hstack((coeff_imp, coeff_env))
    #coeff_eo_in_lo = reduce(np.dot, (coeff_lo_in_ao.T, ovlp_ao, coeff_eo_in_ao))

    return coeff_eo_in_ao


class Embedding():
    def __init__(self, mf, frgm_list, TOL, option=0, nelectrons=None):
        mol = mf.mol
        from pydmet.dmet_tda import build_lo
        from pydmet import mol_lo_tools
        self.coeff_lo_in_ao = build_lo(mol, TOL)

        lo_idx_list = mol_lo_tools.partition_lo_to_imps(
            frgm_list, mol=mol, coeff_ao_lo=self.coeff_lo_in_ao,
            min_weight=0.8
        )

        self.lo_idx = lo_idx_list
        self.imp_lo_idx = lo_idx_list[0]
        self.env_lo_idx = lo_idx_list[1]

        if option == 0:
            self.emb_basis_dmet(mf)
        else:
            self.emb_basis_pod(mf, nelectrons, option)


    def emb_basis_dmet(self, mf, thresh=1e-12):
        ovlp_ao = mf.get_ovlp()
        dm_ao   = mf.make_rdm1()
        from pydmet.rhf import transform_dm_ao_to_lo
        dm_lo   = transform_dm_ao_to_lo(self.coeff_lo_in_ao, dm_ao, ovlp_ao)

        coeff_lo_in_ao = self.coeff_lo_in_ao
        self.coeff_eo_in_ao = emb_basis_dmet(coeff_lo_in_ao, dm_lo, self.lo_idx, thresh)

        coeff_eo_in_lo = np.einsum('ml,mn,np->lp', coeff_lo_in_ao, ovlp_ao, self.coeff_eo_in_ao)
        dm_eo = np.einsum('lp,ln,nq->pq', coeff_eo_in_lo, dm_lo, coeff_eo_in_lo)
        nelec = np.diag(dm_eo).sum()
        nelec = np.round(nelec)
        self.nelec = int(nelec)


    def emb_basis_pod(self, mf, nelectrons, direction=1):
        from pydmet.orbital_projection import get_projection_diabatization
        fock_ao = mf.get_fock()
        ovlp = mf.get_ovlp()
        pod_imp, pod_env = get_projection_diabatization(fock_ao,
                                                        self.coeff_lo_in_ao,
                                                        ovlp, self.lo_idx,
                                                        nelectrons,
                                                        direction=direction)

        self.coeff_eo_in_ao = np.concatenate((pod_imp[0], pod_env[1]), axis=1)
        self.nelec = nelectrons[0]*2


    def get_eomf(self, mf):
        nelec = self.nelec
        print('nelec in EO =', nelec)
        neleca, nelecb = nelec//2, nelec//2 # restricted

        ovlp_ao = mf.get_ovlp()
        #hcore_ao = mf.get_hcore()
        fock_ao  = mf.get_fock()

        coeff_eo_in_ao = self.coeff_eo_in_ao
        # (C_eo *  C_eo^T) * S as the projector from the right
        proj = np.einsum('mi,ni,nl->ml', coeff_eo_in_ao, coeff_eo_in_ao, ovlp_ao)
        fock_ao = np.einsum('nm,nl,ls->ms', proj, fock_ao, proj)

        from scipy.linalg import eigh
        mo_energy, mo_coeff = eigh(fock_ao, ovlp_ao)
        zero_list = np.where(abs(mo_energy) < 10 ** (-7))[0]
        mo_energy = np.delete(mo_energy, zero_list, axis=0)
        mo_coeff = np.delete(mo_coeff, zero_list, axis=1)
        mo_occ = np.zeros_like(mo_energy)
        for i in range(neleca):
            mo_occ[i] = 2

        print_matrix('mo_energy:', mo_energy)

        mol = mf.mol.copy()
        mol.nelectron = nelec # change effective electrons

        eomf = mf.copy()
        eomf.mo_coeff0 = mf.mo_coeff # full system orbitals for dft
        eomf.mo_occ0 = mf.mo_occ # full system orbitals for dft
        # use effective embedding orbitals
        eomf.mo_coeff = mo_coeff
        eomf.mo_occ = mo_occ
        eomf.mo_energy = mo_energy

        return eomf
