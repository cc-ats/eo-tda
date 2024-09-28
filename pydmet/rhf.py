from functools import reduce
import numpy
import pyscf
from pyscf.scf.hf import dot_eri_dm

def get_eri(mf, orbs=None, aosym=4, dataname="eri"):
    """Get the 2-electron integrals in the given basis.

    Args:
        coeffs : the coefficients of the given basis
    
    Returns:
        eri : the 2-electron integrals in the given basis
    """
    eri      = None
    eri_file = None # self.eri_file

    if orbs is None:
        eri = mf._eri
    else:
        if isinstance(mf.mol, pyscf.gto.Mole):
            eri = pyscf.ao2mo.kernel(
                mf.mol, orbs, aosym=aosym, 
                dataname=dataname,
                verbose=mf.verbose
                )

        else:
            assert mf._eri is not None
            eri = pyscf.ao2mo.kernel(
                mf._eri, orbs, aosym=aosym, 
                verbose=mf.verbose
                )
    
    assert eri is not None
    return eri

def transform_dm_ao_to_lo(coeff_ao_lo, dm_ao, ovlp_ao=None):
    if ovlp_ao is not None:
        dm_lo = reduce(numpy.dot, (coeff_ao_lo.T, ovlp_ao, dm_ao, ovlp_ao, coeff_ao_lo))
    else:
        dm_lo = reduce(numpy.dot, (coeff_ao_lo.T, dm_ao, coeff_ao_lo))
    return dm_lo

def make_emb_basis(mf, imp_lo_idx, env_lo_idx, coeff_ao_lo):

    nlo_imp = len(imp_lo_idx)
    nlo_env = len(env_lo_idx)

    ovlp_ao     = mf.get_ovlp()
    dm_ll_ao    = mf.make_rdm1()
    dm_ll_lo    = transform_dm_ao_to_lo(coeff_ao_lo, dm_ll_ao, ovlp_ao)
    nao, nlo    = coeff_ao_lo.shape

    imp_imp_lo_ix = numpy.ix_(imp_lo_idx, imp_lo_idx)
    env_env_lo_ix = numpy.ix_(env_lo_idx, env_lo_idx)
    imp_env_lo_ix = numpy.ix_(imp_lo_idx, env_lo_idx)

    dm_imp_imp_lo = dm_ll_lo[imp_imp_lo_ix]
    dm_env_env_lo = dm_ll_lo[env_env_lo_ix]
    dm_imp_env_lo = dm_ll_lo[imp_env_lo_ix]
    
    assert dm_imp_imp_lo.shape == (nlo_imp, nlo_imp)
    assert dm_env_env_lo.shape == (nlo_env, nlo_env)
    assert dm_imp_env_lo.shape == (nlo_imp, nlo_env)

    uf, sf, vhf = numpy.linalg.svd(dm_imp_env_lo, full_matrices=True)
    coeff_lo_eo_imp_imp  = numpy.eye(nlo_imp)
    #coeff_lo_eo_env_bath = vh.T
    #coeff_lo_eo_env_bath = vhf[:numpy.shape(coeff_lo_eo_imp_imp)[0],:].T
    #coeff_lo_eo_env_env = vhf[numpy.shape(coeff_lo_eo_imp_imp)[0]:,:].T
    
    index_s = numpy.where(sf>0)[0]
    coeff_lo_eo_env_bath = vhf[index_s,:].T
    coeff_lo_eo_env_env = numpy.delete(vhf, index_s, axis=0).T
    
    coeff_ao_lo_imp = coeff_ao_lo[:, imp_lo_idx]
    coeff_ao_lo_env = coeff_ao_lo[:, env_lo_idx]
    coeff_ao_eo_imp  = numpy.dot(coeff_ao_lo_imp, coeff_lo_eo_imp_imp)
    coeff_ao_eo_bath = numpy.dot(coeff_ao_lo_env, coeff_lo_eo_env_bath)
    coeff_ao_eo_env = numpy.dot(coeff_ao_lo_env, coeff_lo_eo_env_env)
    coeff_ao_eo = numpy.hstack((coeff_ao_eo_imp, coeff_ao_eo_bath))
    coeff_lo_eo = reduce(numpy.dot, (coeff_ao_lo.T, ovlp_ao, coeff_ao_eo))

    neo = coeff_ao_eo.shape[1]
    nlo_imp  = coeff_ao_eo_imp.shape[1]
    nlo_env  = coeff_ao_lo_env.shape[1]
    neo_imp  = coeff_ao_eo_imp.shape[1]
    neo_bath = coeff_ao_eo_bath.shape[1]
    assert coeff_ao_eo.shape == (nao, neo)
    assert coeff_lo_eo.shape == (nlo, neo)

    from pydmet.embedding import EmbeddingBasis
    emb_basis = EmbeddingBasis()
    emb_basis.nlo = nlo
    emb_basis.nao = nao
    emb_basis.neo = neo
    emb_basis.neo_imp = neo_imp
    emb_basis.neo_bath = neo_bath
    emb_basis.imp_eo_idx  = range(0, neo_imp)
    emb_basis.bath_eo_idx = range(neo_imp, neo)
    emb_basis.coeff_ao_eo = coeff_ao_eo
    emb_basis.coeff_lo_eo = coeff_lo_eo
    emb_basis.coeff_ao_eo_imp  = coeff_ao_eo_imp
    emb_basis.coeff_ao_eo_bath = coeff_ao_eo_bath
    emb_basis.coeff_ao_eo_env  = coeff_ao_eo_env
    emb_basis.dm_ll_ao = dm_ll_ao
    emb_basis.dm_ll_lo = dm_ll_lo

    return emb_basis
    
def make_new_emb_basis(mf, imp_lo_idx, env_lo_idx, coeff_ao_lo):

    nlo_imp = len(imp_lo_idx)
    nlo_env = len(env_lo_idx)

    ovlp_ao     = mf.get_ovlp()
    dm_ll_ao = mf.make_rdm1()
    dm_ll_lo    = transform_dm_ao_to_lo(coeff_ao_lo, dm_ll_ao, ovlp_ao)
    imp_env_lo_ix = numpy.ix_(imp_lo_idx, env_lo_idx)
    dm_imp_env_lo = dm_ll_lo[imp_env_lo_ix]
    u, s, vh = numpy.linalg.svd(dm_imp_env_lo, full_matrices=False)
    s0 = 20
    '''
    for i in range(len(s)-1):
        if s[i] / s[i+1] > 1e5 :
            s0 = len(s) - i - 1 
            break
    print(f'\n{s0:3.0f} low entanglement LO in impurity get from 1-RDM')
    '''
    mo_coeff    = mf.mo_coeff
    mo_occ      = mf.mo_occ
    nelec = mf.mol.nelectron // 2
    occidx      = numpy.where(mo_occ==2)[0]
    viridx      = numpy.where(mo_occ==0)[0]
    nocc        = len(occidx)
    nvir        = len(viridx)
    if nocc > nvir :
        print(f'nocc = {nocc}, nvir = {nvir}, try to remove homo to get embedding basis')
        extra_dm  = 2 * numpy.einsum('i,j->ij', mo_coeff[:,nelec-1], mo_coeff[:,nelec-1], optimize=True)
        dm_ll_ao -= extra_dm
    else :
        print(f'nocc = {nocc}, nvir = {nvir}, try to add {s0} virtual MO to get embedding basis')
        for i in range(s0):
            print(f'No. {i+1:3d} virtual orbitial with energy {mf.mo_energy[nelec+i]:10.8f} Hartree was added to 1-RDM')
        extra_dm  = 2 * numpy.einsum('ik,jk->ij', mo_coeff[:,nelec:nelec+s0], mo_coeff[:,nelec:nelec+s0], optimize=True)
        dm_ll_ao += extra_dm
    dm_ll_lo    = transform_dm_ao_to_lo(coeff_ao_lo, dm_ll_ao, ovlp_ao)
    nao, nlo    = coeff_ao_lo.shape

    imp_imp_lo_ix = numpy.ix_(imp_lo_idx, imp_lo_idx)
    env_env_lo_ix = numpy.ix_(env_lo_idx, env_lo_idx)
    imp_env_lo_ix = numpy.ix_(imp_lo_idx, env_lo_idx)

    dm_imp_imp_lo = dm_ll_lo[imp_imp_lo_ix]
    dm_env_env_lo = dm_ll_lo[env_env_lo_ix]
    dm_imp_env_lo = dm_ll_lo[imp_env_lo_ix]
    
    assert dm_imp_imp_lo.shape == (nlo_imp, nlo_imp)
    assert dm_env_env_lo.shape == (nlo_env, nlo_env)
    assert dm_imp_env_lo.shape == (nlo_imp, nlo_env)

    uf, sf, vhf = numpy.linalg.svd(dm_imp_env_lo, full_matrices=True)
    coeff_lo_eo_imp_imp  = numpy.eye(nlo_imp)
    print(f'sum of singlar value: {numpy.sum(sf):.8f}')
    index_s = numpy.where(sf>0)[0]
    coeff_lo_eo_env_bath = vhf[index_s,:].T
    coeff_lo_eo_env_env = numpy.delete(vhf, index_s, axis=0).T
    
    coeff_ao_lo_imp = coeff_ao_lo[:, imp_lo_idx]
    coeff_ao_lo_env = coeff_ao_lo[:, env_lo_idx]
    coeff_ao_eo_imp  = numpy.dot(coeff_ao_lo_imp, coeff_lo_eo_imp_imp)
    coeff_ao_eo_bath = numpy.dot(coeff_ao_lo_env, coeff_lo_eo_env_bath)
    coeff_ao_eo_env = numpy.dot(coeff_ao_lo_env, coeff_lo_eo_env_env)
    coeff_ao_eo = numpy.hstack((coeff_ao_eo_imp, coeff_ao_eo_bath))
    coeff_lo_eo = reduce(numpy.dot, (coeff_ao_lo.T, ovlp_ao, coeff_ao_eo))

    neo = coeff_ao_eo.shape[1]
    nlo_imp  = coeff_ao_eo_imp.shape[1]
    nlo_env  = coeff_ao_lo_env.shape[1]
    neo_imp  = coeff_ao_eo_imp.shape[1]
    neo_bath = coeff_ao_eo_bath.shape[1]
    assert coeff_ao_eo.shape == (nao, neo)
    assert coeff_lo_eo.shape == (nlo, neo)

    if nocc > nvir : dm_ll_ao += extra_dm
    else : dm_ll_ao -= extra_dm
    dm_ll_lo    = transform_dm_ao_to_lo(coeff_ao_lo, dm_ll_ao, ovlp_ao)

    from pydmet.embedding import EmbeddingBasis
    emb_basis = EmbeddingBasis()
    emb_basis.nlo = nlo
    emb_basis.nao = nao
    emb_basis.neo = neo
    emb_basis.neo_imp = neo_imp
    emb_basis.neo_bath = neo_bath
    emb_basis.imp_eo_idx  = range(0, neo_imp)
    emb_basis.bath_eo_idx = range(neo_imp, neo)
    emb_basis.coeff_ao_eo = coeff_ao_eo
    emb_basis.coeff_lo_eo = coeff_lo_eo
    emb_basis.coeff_ao_eo_imp  = coeff_ao_eo_imp
    emb_basis.coeff_ao_eo_bath = coeff_ao_eo_bath
    emb_basis.coeff_ao_eo_env  = coeff_ao_eo_env
    emb_basis.dm_ll_ao = dm_ll_ao
    emb_basis.dm_ll_lo = dm_ll_lo
    emb_basis.ovlp_ao  = ovlp_ao

    return emb_basis

def make_emb_prob(mf, emb_basis=None):
    
    dm_ll_ao = emb_basis.dm_ll_ao
    dm_ll_lo = emb_basis.dm_ll_lo
    
    assert dm_ll_lo is not None
    assert dm_ll_ao is not None

    neo      = emb_basis.neo
    neo_imp  = emb_basis.neo_imp
    neo_bath = emb_basis.neo_bath
    imp_eo_idx  = emb_basis.imp_eo_idx
    bath_eo_idx = emb_basis.bath_eo_idx

    coeff_ao_eo = emb_basis.coeff_ao_eo
    coeff_lo_eo = emb_basis.coeff_lo_eo

    hcore_ao    = mf.get_hcore()
    fock_ao     = mf.get_fock(h1e=hcore_ao, dm=dm_ll_ao)

    dm_ll_eo       = reduce(numpy.dot, (coeff_lo_eo.T, dm_ll_lo, coeff_lo_eo))
    print('size in imp+bath =', numpy.shape(dm_ll_eo))

    f1e_eo   = reduce(numpy.dot, (coeff_ao_eo.T, fock_ao,  coeff_ao_eo))
    hcore_eo = reduce(numpy.dot, (coeff_ao_eo.T, hcore_ao,  coeff_ao_eo))
    id_imp  = numpy.zeros((neo, neo))
    id_imp[imp_eo_idx, imp_eo_idx] = 1.0
    nelec = numpy.einsum('ii->', dm_ll_eo)
    nelec = numpy.round(nelec)
    nelec = int(nelec)
    print('nelec in EO =',nelec)

    assert nelec % 2 == 0
    nelecs = (nelec // 2, nelec // 2)
    
    from pydmet.embedding import EmbeddingProblem
    emb_prob = EmbeddingProblem()
    emb_prob.neo    = neo
    emb_prob.neo_imp  = neo_imp
    emb_prob.neo_bath = neo_bath
    emb_prob.imp_eo_idx = imp_eo_idx
    emb_prob.bath_eo_idx = bath_eo_idx
    emb_prob.nelecs = nelecs
    emb_prob.dm0    = dm_ll_eo
    emb_prob.id_imp = id_imp
    emb_prob.coeff_ao_eo = emb_basis.coeff_ao_eo
    emb_prob.coeff_lo_eo = emb_basis.coeff_lo_eo
    emb_prob.hcore_eo = hcore_eo
    emb_prob.f1e  = f1e_eo
    emb_prob.aomf = mf
    '''
    eri_eo      = get_eri(mf, orbs=coeff_ao_eo)
    eri_eo_full = pyscf.ao2mo.restore(1, eri_eo, neo)
    j1e_eo, k1e_eo = dot_eri_dm(eri_eo_full, dm_ll_eo, hermi=1, with_j=True, with_k=True)
    h1e_eo  = f1e_eo - (j1e_eo - k1e_eo * 0.5)
    emb_prob.h1e  = h1e_eo
    emb_prob.h2e  = eri_eo_full
    '''
    return emb_prob
