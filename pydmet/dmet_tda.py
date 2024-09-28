import os, sys
import numpy
import pyscf
from pyscf.lib import logger
from pyscf import gto, scf, lib, dft
from functools import reduce

def full_mol(atoms, charge, xcf='hf', basis='sto3g', TOL=1e-6, verbose=0):
    mol = pyscf.gto.Mole()
    mol.build(
        atom = atoms,
        basis = basis,
        verbose=0,
        charge=charge
        )
    mf = pyscf.scf.RKS(mol)
    mf.verbose = verbose
    mf.max_cycle = 200
    mf.conv_tol  = TOL
    mf.conv_tol_grad = TOL
    mf.xc = xcf
    mf.kernel()

    print('full nelec =', mol.nelectron)
    print('full AO shape =', numpy.shape(mf.mo_coeff))
    return mf

def gas_mol(atomimp, charge, xcf='hf', basis='sto3g', TOL=1e-6, verbose=0):
    mol = pyscf.gto.Mole()
    mol.build(
        atom = atomimp,
        basis = basis,
        verbose=0,
        charge=charge
        )
    mf = pyscf.scf.RKS(mol)
    mf.verbose = verbose
    mf.max_cycle = 200
    mf.conv_tol  = TOL
    mf.conv_tol_grad = TOL
    mf.xc = xcf
    mf.kernel()
    
    print('gas nelec =', mol.nelectron)
    print('gas AO shape =', numpy.shape(mf.mo_coeff))
    return mf

def build_lo(mf, TOL=1e-8):
    from pyscf import lo
    coeff_ao_lo = lo.orth_ao(mf, 'meta-lowdin')
    '''
    coeff_ao_lo = None
    pm = lo.PM(mf.mol, mf.mo_coeff)
    pm.conv_tol = TOL
    pm.dump_flags(verbose=4)
    coeff_ao_lo = pm.kernel(verbose=4)
    '''
    return coeff_ao_lo

def _gen_rhf_response(mf, mo_coeff=None, mo_occ=None,
                      singlet=None, hermi=0, max_memory=None):
    from pyscf.scf import hf, rohf, uhf, ghf, dhf
    '''Generate a function to compute the product of RHF response function and
    RHF density matrices.

    Kwargs:
        singlet (None or boolean) : If singlet is None, response function for
            orbital hessian or CPHF will be generated. If singlet is boolean,
            it is used in TDDFT response kernel.
    '''
    assert (not isinstance(mf, (uhf.UHF, rohf.ROHF)))
    mf.verbose=0
    scf.hf.RHF.get_jk = scf.hf.SCF.get_jk
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    if mf.term == 'eo' :
        mo_coeff0 = mf.mo_coeff0 # needed for dft in _gen_rhf_response
        mo_occ0   = mf.mo_occ0  # needed for dft in _gen_rhf_response
    else :
        mo_coeff0 = mf.mo_coeff # needed for dft in _gen_rhf_response
        mo_occ0   = mf.mo_occ  # needed for dft in _gen_rhf_response
    mol = mf.mol
    if isinstance(mf, hf.KohnShamDFT):
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        if mf.nlc or ni.libxc.is_nlc(mf.xc):
            logger.warn(mf, 'NLC functional found in DFT object.  Its second '
                        'deriviative is not available. Its contribution is '
                        'not included in the response function.')
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
        hybrid = ni.libxc.is_hybrid_xc(mf.xc)

        # mf can be pbc.dft.RKS object with multigrid
        if (not hybrid and
            'MultiGridFFTDF' == getattr(mf, 'with_df', None).__class__.__name__):
            from pyscf.pbc.dft import multigrid
            dm0 = mf.make_rdm1(mo_coeff0, mo_occ0)
            return multigrid._gen_rhf_response(mf, dm0, singlet, hermi)

        # use only singlet or triplet case, removed ground state hessian
        # use original mos
        rho0, vxc, fxc = ni.cache_xc_kernel(mol, mf.grids, mf.xc,
                                            mo_coeff0, mo_occ0, spin=1)
        dm0 = None  #mf.make_rdm1(mo_coeff0, mo_occ0)
        # this ends the modification

        if max_memory is None:
            mem_now = lib.current_memory()[0]
            max_memory = max(2000, mf.max_memory*.8-mem_now)

        if singlet is None:
            # Without specify singlet, used in ground state orbital hessian
            def vind(dm1):
                # The singlet hessian
                if hermi == 2:
                    v1 = numpy.zeros_like(dm1)
                else:
                    v1 = ni.nr_rks_fxc(mol, mf.grids, mf.xc, dm0, dm1, 0, hermi,
                                       rho0, vxc, fxc, max_memory=max_memory)
                if hybrid:
                    if hermi != 2:
                        vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
                        vk *= hyb
                        if abs(omega) > 1e-10:  # For range separated Coulomb
                            vk += mf.get_k(mol, dm1, hermi, omega) * (alpha-hyb)
                        v1 += vj - .5 * vk
                    else:
                        v1 -= .5 * hyb * mf.get_k(mol, dm1, hermi=hermi)
                elif hermi != 2:
                    v1 += mf.get_j(mol, dm1, hermi=hermi)
                return v1

        elif singlet:
            def vind(dm1):
                if hermi == 2:
                    v1 = numpy.zeros_like(dm1)
                else:
                    # nr_rks_fxc_st requires alpha of dm1, dm1*.5 should be scaled
                    v1 = ni.nr_rks_fxc_st(mol, mf.grids, mf.xc, dm0, dm1, 0, True,
                                          rho0, vxc, fxc, max_memory=max_memory)
                    v1 *= .5
                if hybrid:
                    if hermi != 2:
                        vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
                        vk *= hyb
                        if abs(omega) > 1e-10:  # For range separated Coulomb
                            vk += mf.get_k(mol, dm1, hermi, omega) * (alpha-hyb)
                        v1 += vj - .5 * vk
                    else:
                        v1 -= .5 * hyb * mf.get_k(mol, dm1, hermi=hermi)
                elif hermi != 2:
                    v1 += mf.get_j(mol, dm1, hermi=hermi)
                return v1
        else:  # triplet
            def vind(dm1):
                if hermi == 2:
                    v1 = numpy.zeros_like(dm1)
                else:
                    # nr_rks_fxc_st requires alpha of dm1, dm1*.5 should be scaled
                    v1 = ni.nr_rks_fxc_st(mol, mf.grids, mf.xc, dm0, dm1, 0, False,
                                          rho0, vxc, fxc, max_memory=max_memory)
                    v1 *= .5
                if hybrid:
                    vk = mf.get_k(mol, dm1, hermi=hermi)
                    vk *= hyb
                    if abs(omega) > 1e-10:  # For range separated Coulomb
                        vk += mf.get_k(mol, dm1, hermi, omega) * (alpha-hyb)
                    v1 += -.5 * vk
                return v1

    else:  # HF
        if (singlet is None or singlet) and hermi != 2:
            def vind(dm1):
                vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
                return vj - .5 * vk
        else:
            def vind(dm1):
                return -.5 * mf.get_k(mol, dm1, hermi=hermi)

    return vind
       
def make_emb_prob(aomf, imp_list, TOL):
    aomol = aomf.mol
    coeff_ao_lo = build_lo(aomf, TOL)
    from pydmet import mol_lo_tools
    imp_lo_idx_list = mol_lo_tools.partition_lo_to_imps(
        imp_list, mol=aomol, coeff_ao_lo=coeff_ao_lo,
        min_weight=0.8
    )

    imp_lo_idx = imp_lo_idx_list[0]
    env_lo_idx = imp_lo_idx_list[1]
    
    from pydmet import rhf
    emb_basis = rhf.make_emb_basis(aomf, imp_lo_idx, env_lo_idx, coeff_ao_lo)
    #emb_basis = rhf.make_new_emb_basis(aomf, imp_lo_idx, env_lo_idx, coeff_ao_lo)
    emb_prob  = rhf.make_emb_prob(aomf, emb_basis)
    return emb_basis, emb_prob

def get_eomf(aomf, imp_list, TOL, verbose=10):
    emb_basis, emb_prob = make_emb_prob(aomf, imp_list, TOL)
    nelecs = emb_prob.nelecs
    neleca, nelecb = nelecs
    assert neleca == nelecb
    mf0 = emb_prob.aomf

    m = gto.M()
    m.nelectron     = neleca + nelecb
    m.spin          = 0
    m.incore_anyway = True
    m.atom          = mf0.mol.atom
    m.basis         = mf0.mol.basis
    m.charge        = mf0.mol.charge
    m.build()
    m.verbose=0

    coeff_ao_eo = emb_prob.coeff_ao_eo
    f1e  = emb_prob.f1e

    s1e_ao  = mf0.get_ovlp()
    f1e_ao  = reduce(numpy.dot, (s1e_ao, coeff_ao_eo, f1e, coeff_ao_eo.T, s1e_ao))
    from scipy.linalg import eigh
    mo_energy, mo_coeff = eigh(f1e_ao,s1e_ao)
    zero_list = numpy.where(abs(mo_energy) < 10 ** (-7))[0]
    mo_energy = numpy.delete(mo_energy, zero_list, axis=0)
    mo_coeff = numpy.delete(mo_coeff, zero_list, axis=1)
    mo_occ = numpy.zeros_like(mo_energy)
    for i in range(neleca):
        mo_occ[i] = 2

    mf = pyscf.scf.RKS(m)
    mf.verbose   = mf0.verbose
    mf.stdout    = mf0.stdout
    mf.conv_tol  = mf0.conv_tol
    mf.max_cycle = mf0.max_cycle
    mf.verbose   = verbose
    mf.mo_occ    = mo_occ
    mf.mo_energy = mo_energy
    mf.mo_coeff  = mo_coeff
    mf.xc        = mf0.xc
    mf.mo_coeff0 = mf0.mo_coeff # needed for dft in _gen_rhf_response
    mf.mo_occ0   = mf0.mo_occ   # needed for dft in _gen_rhf_response

    return emb_basis, emb_prob, mf

def solve_tda(mf, nstates, term, verbose=10):
    mf.verbose = verbose
    mf.term = term
    scf.hf.RHF.gen_response = _gen_rhf_response
    from pyscf.tdscf.rhf import TDA as tda
    tdobj = tda(mf)
    result = tdobj.kernel(nstates=nstates)
    tda.analyze(tdobj, verbose=verbose)
    excited_energy, amplitude = result
    #from pydmet import analysis_tool
    #analysis_tool.get_nto_fig(mf, tdobj, excited_energy)
    return excited_energy, amplitude

def runtda(atom, atomimp, charge, imp_list, xcf='hf', basis='sto3g', nstates=3, tol=1e-5, verbose=0):
    ''' restricted DMET-TDA

    Args:    
        
        atom : str or list
            atoms coordinate for full system.
        atomimp : str or list
            atoms coordinate for impurity.
        charge : int
            system charge.
        imp_list : list
            impurity atoms list.
        xcf : str
            exchange-correlation function used in SCF and TDA calculation.
        basis : str
            basis used in SCF and TDA calculation.
        nstates : int
            number of state will calculation in TDA, default is 3.
        tol : int
            SCF convergence tol for energy change and energy graident, default is 1e-5.
    '''
    TOL = os.environ.get("TOL", tol)
    
    gasmf = gas_mol(atomimp, charge, xcf, basis, TOL, verbose)
    ene_gas_tda, amp_gas_tda = solve_tda(gasmf, nstates, term='gas')

    aomf = full_mol(atom, charge, xcf, basis, TOL, verbose)
    ene_ao_tda, amp_ao_tda = solve_tda(aomf, nstates, term='ao')

    emb_basis, emb_prob, eomf = get_eomf(aomf, imp_list, TOL)
    ene_eo_tda, amp_eo_tda = solve_tda(eomf, nstates, term='eo')
    
    from pydmet import analysis_tool
    analysis_tool.output(gasmf, aomf, eomf, emb_basis, amp_gas_tda, amp_ao_tda, amp_eo_tda, ene_gas_tda, ene_ao_tda, ene_eo_tda)
