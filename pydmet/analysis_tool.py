import numpy
from pyscf.lib import logger
from pyscf import scf, lib
from functools import reduce

def get_tdm(mf, amplitude):
    occidx = numpy.where(mf.mo_occ==2)[0]
    viridx = numpy.where(mf.mo_occ==0)[0]
    orbv = mf.mo_coeff[:,viridx]
    orbo = mf.mo_coeff[:,occidx]
    tdm  = []
    for i in range(len(amplitude)):
        tdm.append(numpy.einsum('ij,jk,lk->il', orbo, amplitude[i][0], orbv, optimize=True))
    return tdm
    
def ie_mo_distribution(aomf, coeff_ao_lo, imp_list, verbose=10):
    ovlp = aomf.get_ovlp()
    mol = aomf.mol
    from pydmet import mol_lo_tools
    imp_lo_idx_list = mol_lo_tools.partition_lo_to_imps(
        imp_list, mol=mol, coeff_ao_lo=coeff_ao_lo,
        min_weight=0.8
    )
    imp_lo_idx = imp_lo_idx_list[0]
    env_lo_idx = imp_lo_idx_list[1]

    imp = imp_list[0]
    occidx = numpy.where(aomf.mo_occ==2)[0]
    viridx = numpy.where(aomf.mo_occ==0)[0]
    mo_coeff_ao = aomf.mo_coeff
    nocc = mol.nelectron // 2
    log = logger.new_logger(verbose=verbose)
    ao_slice_by_atom = mol.aoslice_by_atom()
    bound = ao_slice_by_atom[imp[len(imp)-1], 3]

    homo_ii_ao  = numpy.einsum('p,pq,q->', mo_coeff_ao[:bound,nocc-1], ovlp[:bound,:bound], mo_coeff_ao[:bound,nocc-1], optimize=True)
    homo_ie_ao  = 2 * numpy.einsum('p,pq,q->', mo_coeff_ao[:bound,nocc-1], ovlp[:bound,bound:], mo_coeff_ao[bound:,nocc-1], optimize=True)
    homo_ee_ao  = numpy.einsum('p,pq,q->', mo_coeff_ao[bound:,nocc-1], ovlp[bound:,bound:], mo_coeff_ao[bound:,nocc-1], optimize=True)
    lumo_ii_ao  = numpy.einsum('p,pq,q->', mo_coeff_ao[:bound,nocc], ovlp[:bound,:bound], mo_coeff_ao[:bound,nocc], optimize=True)
    lumo_ie_ao  = 2 * numpy.einsum('p,pq,q->', mo_coeff_ao[:bound,nocc], ovlp[:bound,bound:], mo_coeff_ao[bound:,nocc], optimize=True)
    lumo_ee_ao  = numpy.einsum('p,pq,q->', mo_coeff_ao[bound:,nocc], ovlp[bound:,bound:], mo_coeff_ao[bound:,nocc], optimize=True)
    lumop_ii_ao = numpy.einsum('p,pq,q->', mo_coeff_ao[:bound,nocc+1], ovlp[:bound,:bound], mo_coeff_ao[:bound,nocc+1], optimize=True)
    lumop_ie_ao = 2 * numpy.einsum('p,pq,q->', mo_coeff_ao[:bound,nocc+1], ovlp[:bound,bound:], mo_coeff_ao[bound:,nocc+1], optimize=True)
    lumop_ee_ao = numpy.einsum('p,pq,q->', mo_coeff_ao[bound:,nocc+1], ovlp[bound:,bound:], mo_coeff_ao[bound:,nocc+1], optimize=True)

    mo_coeff_lo = numpy.einsum('ji,jk,kl->il', coeff_ao_lo, ovlp, mo_coeff_ao, optimize=True)
    ovlp_lo = numpy.einsum('ji,jk,kl->il', coeff_ao_lo, ovlp, coeff_ao_lo, optimize=True)

    homo_ii_lo  = numpy.einsum('p,pq,q->', mo_coeff_lo[imp_lo_idx][:,nocc-1], ovlp_lo[imp_lo_idx,:][:,imp_lo_idx], mo_coeff_lo[imp_lo_idx][:,nocc-1], optimize=True)
    homo_ie_lo  = numpy.einsum('p,pq,q->', mo_coeff_lo[imp_lo_idx][:,nocc-1], ovlp_lo[imp_lo_idx,:][:,env_lo_idx], mo_coeff_lo[env_lo_idx][:,nocc-1], optimize=True) * 2
    homo_ee_lo  = numpy.einsum('p,pq,q->', mo_coeff_lo[env_lo_idx][:,nocc-1], ovlp_lo[env_lo_idx,:][:,env_lo_idx], mo_coeff_lo[env_lo_idx][:,nocc-1], optimize=True)
    lumo_ii_lo  = numpy.einsum('p,pq,q->', mo_coeff_lo[imp_lo_idx][:,nocc], ovlp_lo[imp_lo_idx,:][:,imp_lo_idx], mo_coeff_lo[imp_lo_idx][:,nocc], optimize=True)
    lumo_ie_lo  = numpy.einsum('p,pq,q->', mo_coeff_lo[imp_lo_idx][:,nocc], ovlp_lo[imp_lo_idx,:][:,env_lo_idx], mo_coeff_lo[env_lo_idx][:,nocc], optimize=True) * 2
    lumo_ee_lo  = numpy.einsum('p,pq,q->', mo_coeff_lo[env_lo_idx][:,nocc], ovlp_lo[env_lo_idx,:][:,env_lo_idx], mo_coeff_lo[env_lo_idx][:,nocc], optimize=True)
    lumop_ii_lo = numpy.einsum('p,pq,q->', mo_coeff_lo[imp_lo_idx][:,nocc+1], ovlp_lo[imp_lo_idx,:][:,imp_lo_idx], mo_coeff_lo[imp_lo_idx][:,nocc+1], optimize=True)
    lumop_ie_lo = numpy.einsum('p,pq,q->', mo_coeff_lo[imp_lo_idx][:,nocc+1], ovlp_lo[imp_lo_idx,:][:,env_lo_idx], mo_coeff_lo[env_lo_idx][:,nocc+1], optimize=True) * 2
    lumop_ee_lo = numpy.einsum('p,pq,q->', mo_coeff_lo[env_lo_idx][:,nocc+1], ovlp_lo[env_lo_idx,:][:,env_lo_idx], mo_coeff_lo[env_lo_idx][:,nocc+1], optimize=True)

    log.debug('MO_AO    i&i      i&e      e&e     sum ')
    log.debug('%5s %8.5f %8.5f %8.5f %7.5f', ' homo', homo_ii_ao, homo_ie_ao, homo_ee_ao, homo_ii_ao + homo_ie_ao + homo_ee_ao)
    log.debug('%5s %8.5f %8.5f %8.5f %7.5f', ' lumo', lumo_ii_ao, lumo_ie_ao, lumo_ee_ao, lumo_ii_ao + lumo_ie_ao + lumo_ee_ao)
    log.debug('%5s %8.5f %8.5f %8.5f %7.5f', 'lumop', lumop_ii_ao, lumop_ie_ao, lumop_ee_ao, lumop_ii_ao + lumop_ie_ao + lumop_ee_ao)
    log.debug('MO_LO    i&i      i&e      e&e     sum ')
    log.debug('%5s %8.5f %8.5f %8.5f %7.5f', ' homo', homo_ii_lo, homo_ie_lo, homo_ee_lo, homo_ii_lo + homo_ie_lo + homo_ee_lo)
    log.debug('%5s %8.5f %8.5f %8.5f %7.5f', ' lumo', lumo_ii_lo, lumo_ie_lo, lumo_ee_lo, lumo_ii_lo + lumo_ie_lo + lumo_ee_lo)
    log.debug('%5s %8.5f %8.5f %8.5f %7.5f', 'lumop', lumop_ii_lo, lumop_ie_lo, lumop_ee_lo, lumop_ii_lo + lumop_ie_lo + lumop_ee_lo)

def get_mopair(mf, emb_basis, mhomo=0, plumo=0):
    mo_energy = mf.mo_energy
    nelec     = mf.mol.nelectron // 2
    homo = mo_energy[nelec - 1 - mhomo]
    lumo = mo_energy[nelec + plumo]
    
    if mf.term == 'gas' :
        return homo, lumo 
    else :
        ovlp = mf.get_ovlp()
        mo_coeff  = mf.mo_coeff
        coeff_ao_eo_imp  = emb_basis.coeff_ao_eo_imp
        coeff_ao_eo_bath = emb_basis.coeff_ao_eo_bath
        coeff_ao_eo_env  = emb_basis.coeff_ao_eo_env
        homo_coeff  = mo_coeff[:,nelec - 1 - mhomo]
        lumo_coeff  = mo_coeff[:,nelec + plumo]
        #from pyscf.tools import cubegen
        #mol = mf.mol
        #cubegen.orbital(mol, f'./cube/frame4homo-{mhomo}'+mf.term+mol.basis+'.cube', homo_coeff)
        #cubegen.orbital(mol, f'./cube/frame4lumo-{plumo}'+mf.term+mol.basis+'.cube', lumo_coeff)

        imp_homo    = numpy.einsum('ji,jk,k->i', coeff_ao_eo_imp, ovlp, homo_coeff, optimize=True)
        bath_homo   = numpy.einsum('ji,jk,k->i', coeff_ao_eo_bath, ovlp, homo_coeff, optimize=True)
        env_homo    = numpy.einsum('ji,jk,k->i', coeff_ao_eo_env, ovlp, homo_coeff, optimize=True)
        imp_lumo    = numpy.einsum('ji,jk,k->i', coeff_ao_eo_imp, ovlp, lumo_coeff, optimize=True)
        bath_lumo   = numpy.einsum('ji,jk,k->i', coeff_ao_eo_bath, ovlp, lumo_coeff, optimize=True)
        env_lumo    = numpy.einsum('ji,jk,k->i', coeff_ao_eo_env, ovlp, lumo_coeff, optimize=True)
        
        w2imphomo   = numpy.einsum('n,n->', imp_homo, imp_homo, optimize=True)
        w2bathhomo  = numpy.einsum('n,n->', bath_homo, bath_homo, optimize=True)
        w2envhomo   = numpy.einsum('n,n->', env_homo, env_homo, optimize=True)
        w2implumo   = numpy.einsum('n,n->', imp_lumo, imp_lumo, optimize=True)
        w2bathlumo  = numpy.einsum('n,n->', bath_lumo, bath_lumo, optimize=True)
        w2envlumo   = numpy.einsum('n,n->', env_lumo, env_lumo, optimize=True)
        w2 = [w2imphomo, w2bathhomo, w2envhomo, w2implumo, w2bathlumo, w2envlumo]
        return homo, lumo, w2   

def output(gasmf, aomf, eomf, emb_basis, amp_gas_tda, amp_ao_tda, amp_eo_tda, egas, eao, eeo, verbose=10):
    log = logger.new_logger(verbose=verbose)
    ovlp_ao = aomf.get_ovlp()
    gashomo, gaslumo     = get_mopair(gasmf, emb_basis, mhomo=0, plumo=0)
    aohomo, aolumo, w2ao = get_mopair(aomf, emb_basis, mhomo=0, plumo=0)
    eohomo, eolumo, w2eo = get_mopair(eomf, emb_basis, mhomo=0, plumo=0)

    log.debug('\nHOMO and LUMO energy for gas, eo, ao')
    log.debug('gas   HOMO: %10.7f Hartree   LUMO: %10.7f Hartree', gashomo, gaslumo)
    log.debug('ao    HOMO: %10.7f Hartree   LUMO: %10.7f Hartree', aohomo, aolumo)
    log.debug('eo    HOMO: %10.7f Hartree   LUMO: %10.7f Hartree', eohomo, eolumo)
    log.debug('eo-ao HOMO: %10.7f eV        LUMO: %10.7f eV', 27.2114 * (eohomo-aohomo), 27.2114 * (eolumo-aolumo))
    log.debug('\nHOMO and LUMO partition for eo, ao on imp, bath, env')
    log.debug('              imp     bat     env     sum')
    log.debug('eo    HOMO: %7.5f %7.5f %7.5f %7.5f', w2eo[0], w2eo[1], w2eo[2], w2eo[0] + w2eo[1] + w2eo[2])
    log.debug('ao    HOMO: %7.5f %7.5f %7.5f %7.5f', w2ao[0], w2ao[1], w2ao[2], w2ao[0] + w2ao[1] + w2ao[2])
    log.debug('eo    LUMO: %7.5f %7.5f %7.5f %7.5f', w2eo[3], w2eo[4], w2eo[5], w2eo[3] + w2eo[4] + w2eo[5])
    log.debug('ao    LUMO: %7.5f %7.5f %7.5f %7.5f', w2ao[3], w2ao[4], w2ao[5], w2ao[3] + w2ao[4] + w2ao[5])
    
    gashomo, gaslumo     = get_mopair(gasmf, emb_basis, mhomo=1, plumo=1)
    aohomo, aolumo, w2ao = get_mopair(aomf, emb_basis, mhomo=1, plumo=1)
    eohomo, eolumo, w2eo = get_mopair(eomf, emb_basis, mhomo=1, plumo=1)

    log.debug('\nHOMO-1 and LUMO+1 energy for gas, eo, ao')
    log.debug('gas   HOMO-1: %10.7f Hartree   LUMO+1: %10.7f Hartree', gashomo, gaslumo)
    log.debug('ao    HOMO-1: %10.7f Hartree   LUMO+1: %10.7f Hartree', aohomo, aolumo)
    log.debug('eo    HOMO-1: %10.7f Hartree   LUMO+1: %10.7f Hartree', eohomo, eolumo)
    log.debug('eo-ao HOMO-1: %10.7f eV        LUMO+1: %10.7f eV', 27.2114 * (eohomo-aohomo), 27.2114 * (eolumo-aolumo))
    log.debug('\nHOMO-1 and LUMO+1 partition for eo, ao on imp, bath, env')
    log.debug('              imp     bat     env     sum')
    log.debug('eo  HOMO-1: %7.5f %7.5f %7.5f %7.5f', w2eo[0], w2eo[1], w2eo[2], w2eo[0] + w2eo[1] + w2eo[2])
    log.debug('ao  HOMO-1: %7.5f %7.5f %7.5f %7.5f', w2ao[0], w2ao[1], w2ao[2], w2ao[0] + w2ao[1] + w2ao[2])
    log.debug('eo  LUMO+1: %7.5f %7.5f %7.5f %7.5f', w2eo[3], w2eo[4], w2eo[5], w2eo[3] + w2eo[4] + w2eo[5])
    log.debug('ao  LUMO+1: %7.5f %7.5f %7.5f %7.5f', w2ao[3], w2ao[4], w2ao[5], w2ao[3] + w2ao[4] + w2ao[5])

    log.debug('\nExcitation energy for each state in gas, eo, ao')
    for i in range(len(eao)):
        log.debug('state %3d: %7.4f eV    %7.4f eV   %7.4f eV', i+1, egas[i] * 27.2114, eeo[i] * 27.2114, eao[i] * 27.2114)
    
    tdm_gas_tda = get_tdm(gasmf, amp_gas_tda)
    tdm_ao = get_tdm(aomf, amp_ao_tda)
    tdm_eo = get_tdm(eomf, amp_eo_tda)

    tdm_gasao = []
    for i in range(len(tdm_gas_tda)):
        tdm_gasaoi = numpy.zeros_like(tdm_ao[i])
        size = numpy.shape(tdm_gas_tda[i])
        tdm_gasaoi[:size[0],:size[1]] = tdm_gas_tda[i]
        tdm_gasao.append(tdm_gasaoi)
                
    print('Tr(R_eoSR_ao.TS) =')
    for i in range(len(tdm_eo)):
        for j in range(len(tdm_ao)):
            print('{:.4f}'.format(abs(2 * numpy.einsum('ij,jk,lk,li->', tdm_eo[i], ovlp_ao, tdm_ao[j], ovlp_ao, optimize=True))),end=' ')
        print('')
        
    print('Tr(R_aoSR_gasao.TS) =')
    for i in range(len(tdm_ao)):
        for j in range(len(tdm_gasao)):
            print('{:.4f}'.format(abs(2 * numpy.einsum('ij,jk,lk,li->', tdm_ao[i], ovlp_ao, tdm_gasao[j], ovlp_ao, optimize=True))),end=' ')
        print('')
        
    print('Tr(R_eoSR_gasao.TS) =')
    for i in range(len(tdm_eo)):
        for j in range(len(tdm_gasao)):
            print('{:.4f}'.format(abs(2 * numpy.einsum('ij,jk,lk,li->', tdm_eo[i], ovlp_ao, tdm_gasao[j], ovlp_ao, optimize=True))),end=' ')
        print('')

def get_nto_fig(mf, tdobj, excited_energy):
    term = mf.term
    nocc = len(numpy.where(mf.mo_occ>0)[0])
    for k in range(len(excited_energy)):
        weights, ntos = tdobj.get_nto(state=k)
        numpy.savetxt(f'./nto/w2-state{k}'+term+'.out', weights, delimiter=" ", fmt="% 20.16f")
        numpy.savetxt(f'./nto/ntos-state{k}'+term+'.out', ntos, delimiter=",", fmt="% 20.16f")
        w_idx = numpy.where(weights>0.01)[0]
        nto_o = numpy.zeros_like(ntos[:,0])
        nto_v = numpy.zeros_like(nto_o)
        for i in w_idx:
            nto_o += weights[i] * ntos[:,i]
            nto_v += weights[i] * ntos[:,i+nocc]
        from pyscf.tools import cubegen
        mol = mf.mol
        cubegen.orbital(mol, './cube/nto_o{}'.format(k)+term+'.cube', nto_o)
        cubegen.orbital(mol, './cube/nto_v{}'.format(k)+term+'.cube', nto_v)

def ie_charge_transfer_number_mul(mf, amplitude, imp_list, verbose=10):
    ovlp = mf.get_ovlp()
    imp = imp_list[0]
    mol = mf.mol
    occidx = numpy.where(mf.mo_occ==2)[0]
    viridx = numpy.where(mf.mo_occ==0)[0]
    log = logger.new_logger(verbose=verbose)
    ao_slice_by_atom = mol.aoslice_by_atom()
    bound = ao_slice_by_atom[imp[len(imp)-1], 3]

    mul_imp_occ = numpy.zeros((len(occidx),len(occidx)))
    mul_imp_vir = numpy.zeros((len(viridx),len(viridx)))
    mul_env_occ = numpy.zeros((len(occidx),len(occidx)))
    mul_env_vir = numpy.zeros((len(viridx),len(viridx)))

    for i in occidx:
        for j in occidx:
            dmij = numpy.dot(mf.mo_coeff[:,[i]], mf.mo_coeff[:,[j]].T)
            mul_imp_occ[i,j] = numpy.einsum('ij,ji->', dmij[:bound,:], ovlp[:,:bound], optimize=True)
            mul_env_occ[i,j] = numpy.einsum('ij,ji->', dmij[bound:,:], ovlp[:,bound:], optimize=True)
    for a in viridx:
        for b in viridx:
            dmab = numpy.dot(mf.mo_coeff[:,[a]], mf.mo_coeff[:,[b]].T)
            mul_imp_vir[a-viridx[0],b-viridx[0]] = numpy.einsum('ij,ji->', dmab[:bound,:], ovlp[:,:bound], optimize=True)
            mul_env_vir[a-viridx[0],b-viridx[0]] = numpy.einsum('ij,ji->', dmab[bound:,:], ovlp[:,bound:], optimize=True)
    log.debug('state imp->imp imp->env env->imp env->env  sum ')
    for i in range(len(amplitude)):
        ii = 2 * numpy.einsum('ia,ab,bj,ji->', amplitude[i][0], mul_imp_vir, amplitude[i][0].T, mul_imp_occ, optimize=True)
        ei = 2 * numpy.einsum('ia,ab,bj,ji->', amplitude[i][0], mul_imp_vir, amplitude[i][0].T, mul_env_occ, optimize=True)
        ie = 2 * numpy.einsum('ia,ab,bj,ji->', amplitude[i][0], mul_env_vir, amplitude[i][0].T, mul_imp_occ, optimize=True)
        ee = 2 * numpy.einsum('ia,ab,bj,ji->', amplitude[i][0], mul_env_vir, amplitude[i][0].T, mul_env_occ, optimize=True)
        log.debug('%3d   %8.6f %8.6f %8.6f %8.6f %5.3f', i, ii, ie, ei, ee, ii+ie+ei+ee)

def ibe_charge_transfer_number_mul(mf, amplitude, emb_basis, verbose=10):
    ovlp = mf.get_ovlp()
    mo   = mf.mo_coeff
    occidx = numpy.where(mf.mo_occ==2)[0]
    viridx = numpy.where(mf.mo_occ==0)[0]
    log = logger.new_logger(verbose=verbose)

    coeff_ao_eo_imp  = emb_basis.coeff_ao_eo_imp
    coeff_ao_eo_bath = emb_basis.coeff_ao_eo_bath
    coeff_ao_eo_env  = emb_basis.coeff_ao_eo_env
    coeff_ao_eo = numpy.hstack((coeff_ao_eo_imp, coeff_ao_eo_bath, coeff_ao_eo_env))
    mo_eo = numpy.einsum('ij,jk,kl->il', coeff_ao_eo.T, ovlp, mo, optimize=True)
    ovlp_eo = numpy.einsum('ij,jk,kl->il', coeff_ao_eo.T, ovlp, coeff_ao_eo, optimize=True)
    bound1 = numpy.shape(coeff_ao_eo_imp)[1]
    bound2 = numpy.shape(coeff_ao_eo_imp)[1] + numpy.shape(coeff_ao_eo_bath)[1]
    
    mul_imp_occ = numpy.zeros((len(occidx),len(occidx)))
    mul_imp_vir = numpy.zeros((len(viridx),len(viridx)))
    mul_bat_occ = numpy.zeros((len(occidx),len(occidx)))
    mul_bat_vir = numpy.zeros((len(viridx),len(viridx)))
    mul_env_occ = numpy.zeros((len(occidx),len(occidx)))
    mul_env_vir = numpy.zeros((len(viridx),len(viridx)))
    for i in occidx:
        for j in occidx:
            dmij = numpy.dot(mo_eo[:,[i]], mo_eo[:,[j]].T)
            mul_imp_occ[i,j] = numpy.einsum('ij,ji->', dmij[:bound1,:], ovlp_eo[:,:bound1], optimize=True)
            mul_bat_occ[i,j] = numpy.einsum('ij,ji->', dmij[bound1:bound2,:], ovlp_eo[:,bound1:bound2], optimize=True)
            mul_env_occ[i,j] = numpy.einsum('ij,ji->', dmij[bound2:,:], ovlp_eo[:,bound2:], optimize=True)
    for a in viridx:
        for b in viridx:
            dmab = numpy.dot(mo_eo[:,[a]], mo_eo[:,[b]].T)
            mul_imp_vir[a-viridx[0],b-viridx[0]] = numpy.einsum('ij,ji->', dmab[:bound1,:], ovlp_eo[:,:bound1], optimize=True)
            mul_bat_vir[a-viridx[0],b-viridx[0]] = numpy.einsum('ij,ji->', dmab[bound1:bound2,:], ovlp_eo[:,bound1:bound2], optimize=True)
            mul_env_vir[a-viridx[0],b-viridx[0]] = numpy.einsum('ij,ji->', dmab[bound2:,:], ovlp_eo[:,bound2:], optimize=True)
    log.debug('state imp->imp imp->bat imp->env bat->imp bat->bat bat->env env->imp env->bat env->env  sum')
    for i in range(len(amplitude)):
        ii = 2 * numpy.einsum('ia,ab,bj,ji->', amplitude[i][0], mul_imp_vir, amplitude[i][0].T, mul_imp_occ, optimize=True)
        bb = 2 * numpy.einsum('ia,ab,bj,ji->', amplitude[i][0], mul_bat_vir, amplitude[i][0].T, mul_bat_occ, optimize=True)
        ee = 2 * numpy.einsum('ia,ab,bj,ji->', amplitude[i][0], mul_env_vir, amplitude[i][0].T, mul_env_occ, optimize=True)
        ei = 2 * numpy.einsum('ia,ab,bj,ji->', amplitude[i][0], mul_imp_vir, amplitude[i][0].T, mul_env_occ, optimize=True)
        eb = 2 * numpy.einsum('ia,ab,bj,ji->', amplitude[i][0], mul_bat_vir, amplitude[i][0].T, mul_env_occ, optimize=True)
        ie = 2 * numpy.einsum('ia,ab,bj,ji->', amplitude[i][0], mul_env_vir, amplitude[i][0].T, mul_imp_occ, optimize=True)
        ib = 2 * numpy.einsum('ia,ab,bj,ji->', amplitude[i][0], mul_bat_vir, amplitude[i][0].T, mul_imp_occ, optimize=True)
        bi = 2 * numpy.einsum('ia,ab,bj,ji->', amplitude[i][0], mul_imp_vir, amplitude[i][0].T, mul_bat_occ, optimize=True)
        be = 2 * numpy.einsum('ia,ab,bj,ji->', amplitude[i][0], mul_env_vir, amplitude[i][0].T, mul_bat_occ, optimize=True)
        total = ii + ib + ie + bi + bb + be + ei + eb + ee
        log.debug('%3d   %8.6f %8.6f %8.6f %8.6f %8.6f %8.6f %8.6f %8.6f %8.6f %5.3f', i, ii, ib, ie, bi, bb, be, ei, eb, ee, total)

def ad_density(mf, amplitude, emb_basis, verbose=10):

    coeff_ao_eo_imp  = emb_basis.coeff_ao_eo_imp
    coeff_ao_eo_bath = emb_basis.coeff_ao_eo_bath
    occidx = numpy.where(mf.mo_occ==2)[0]
    viridx = numpy.where(mf.mo_occ==0)[0]
    orbv = mf.mo_coeff[:,viridx]
    orbo = mf.mo_coeff[:,occidx]
    ovlp = mf.get_ovlp()
    coeff_t_eo = numpy.hstack((coeff_ao_eo_imp, coeff_ao_eo_bath))
    log = logger.new_logger(verbose=verbose)
    log.debug('state Tr(CSASC) Tr(CSDSC)')
    for i in range(len(amplitude)):
        vxxvi = numpy.einsum('ij,kj,kl,ml->im', orbv, amplitude[i][0], amplitude[i][0], orbv, optimize=True)
        oxxoi = numpy.einsum('ij,jk,lk,ml->im', orbo, amplitude[i][0], amplitude[i][0], orbo, optimize=True)
        tra = 2 * numpy.einsum('ij,jk,kl,lm,mi->', coeff_t_eo.T, ovlp, vxxvi, ovlp, coeff_t_eo, optimize=True)
        trd = 2 * numpy.einsum('ij,jk,kl,lm,mi->', coeff_t_eo.T, ovlp, oxxoi, ovlp, coeff_t_eo, optimize=True)
        log.debug('%3d    %7.5f   %7.5f', i, tra, trd)

def energy_decomponent(mf, energy, amplitude, verbose=10):

    def get_tda_f1e(mf):
        ni = mf._numint
        mf.verbose=0
        if mf.term == 'eo' :
            mo_coeff0 = mf.mo_coeff0 # needed for dft in _gen_rhf_response
            mo_occ0   = mf.mo_occ0  # needed for dft in _gen_rhf_response
        else :
            mo_coeff0 = mf.mo_coeff # needed for dft in _gen_rhf_response
            mo_occ0   = mf.mo_occ  # needed for dft in _gen_rhf_response
        dm0 = mf.make_rdm1(mo_coeff0, mo_occ0)
        n, exc, vf = ni.nr_rks(mf.mol, mf.grids, mf.xc, dm0)
        return vf
    
    def get_tda_f2e(mf, dm1):
        ni = mf._numint
        mf.verbose=0
        if mf.term == 'eo' :
            mo_coeff0 = mf.mo_coeff0 # needed for dft in _gen_rhf_response
            mo_occ0   = mf.mo_occ0  # needed for dft in _gen_rhf_response
        else :
            mo_coeff0 = mf.mo_coeff # needed for dft in _gen_rhf_response
            mo_occ0   = mf.mo_occ  # needed for dft in _gen_rhf_response
        mol = mf.mol
        dm0 = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
        rho0, vxc, fxc = ni.cache_xc_kernel(mol, mf.grids, mf.xc,
                                            mo_coeff0, mo_occ0, spin=1)
        vf = 0.5 * ni.nr_rks_fxc_st(mol, mf.grids, mf.xc, dm0, dm1, 0, singlet=True,
                                    rho0=rho0, vxc=vxc, fxc=fxc)
        return vf
    
    def get_tda_jk(mf, dm1):
        ni = mf._numint
        mf.verbose=0
        scf.hf.RHF.get_jk = scf.hf.SCF.get_jk
        mol = mf.mol
        hybrid = ni.libxc.is_hybrid_xc(mf.xc)
        if hybrid:
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
            vj, vk = mf.get_jk(mol, dm1, hermi=0)
            vk *= hyb
            if abs(omega) > 1e-10:  # For range separated Coulomb
                vk += mf.get_k(mol, dm1, 0, omega) * (alpha-hyb)
        elif mf.xc == 'hf':
            vj, vk = mf.get_jk(mol, dm1, hermi=0)
        else :
            vj = mf.get_j(mol, dm1, hermi=0)
            vk = numpy.zeros_like(vj)
        return vj, 0.5 * vk
    
    def get_1e(mf):

        def get_ia(mf, matrix_ao):
            mo_occ = mf.mo_occ
            mo_coeff = mf.mo_coeff
            occidx = numpy.where(mo_occ==2)[0]
            viridx = numpy.where(mo_occ==0)[0]
            matrix_mo = reduce(numpy.dot, (mo_coeff.conj().T, matrix_ao, mo_coeff))
            moo = matrix_mo[occidx[:,None],occidx]
            mvv = matrix_mo[viridx[:,None],viridx]
            m_ia = mvv.diagonal() - moo.diagonal()[:,None]
            return m_ia
        
        vj, vk = get_tda_jk(mf, mf.make_rdm1(mf.mo_coeff, mf.mo_occ))
        vf = get_tda_f1e(mf)
        h_ia = get_ia(mf, mf.get_hcore())
        j_ia = get_ia(mf, vj)
        k_ia = get_ia(mf, vk)
        f_ia = get_ia(mf, vf)

        return h_ia, j_ia, k_ia, f_ia

    def get_2e(mf, amplitude):
        occidx = numpy.where(mf.mo_occ==2)[0]
        viridx = numpy.where(mf.mo_occ==0)[0]
        orbv = mf.mo_coeff[:,viridx]
        orbo = mf.mo_coeff[:,occidx]
        dmov = lib.einsum('ov,qv,po->pq', amplitude*2, orbv.conj(), orbo)
        vjao, vkao = get_tda_jk(mf, dmov)
        vfao = get_tda_f2e(mf, dmov)
        vjov = lib.einsum('pq,po,qv->ov', vjao, orbo.conj(), orbv)
        vkov = lib.einsum('pq,po,qv->ov', vkao, orbo.conj(), orbv)
        vfov = lib.einsum('pq,po,qv->ov', vfao, orbo.conj(), orbv)
        return vjov, vkov, vfov
    
    log = logger.new_logger(verbose=verbose)
    log.debug('TDA energy decomponent for %3s with %6s', mf.term, mf.xc)
    for i in range(len(amplitude)):
        h_ia, j_ia, k_ia, f_ia = get_1e(mf)
        w1h =   2 * numpy.einsum('ij,ij->', amplitude[i][0] ** 2, h_ia, optimize=True) * 27.2114
        w1j =   2 * numpy.einsum('ij,ij->', amplitude[i][0] ** 2, j_ia, optimize=True) * 27.2114
        w1k = - 2 * numpy.einsum('ij,ij->', amplitude[i][0] ** 2, k_ia, optimize=True) * 27.2114
        w1f =   2 * numpy.einsum('ij,ij->', amplitude[i][0] ** 2, f_ia, optimize=True) * 27.2114
        vjov, vkov, vfov = get_2e(mf, amplitude[i][0])
        w2j =   2 * numpy.einsum('ij,ij->', amplitude[i][0], vjov, optimize=True) * 27.2114
        w2k = - 2 * numpy.einsum('ij,ij->', amplitude[i][0], vkov, optimize=True) * 27.2114
        w2f =   2 * numpy.einsum('ij,ij->', amplitude[i][0], vfov, optimize=True) * 27.2114
        error = energy[i] * 27.2114 - (w1h+w1j+w1k+w1f+w2j+w2k+w2f)
        log.debug('w1h = %7.3f eV, w1j = %7.3f eV, w1k = %7.3f eV, w1f = %7.3f eV, w2j = %7.3f eV, w2k = %7.3f eV, w2f = %7.3f eV, error = %7.3f eV', w1h, w1j, w1k, w1f, w2j, w2k, w2f, error)
  