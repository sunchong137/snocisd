from pyscf import gto, scf, ci
from noci_jax import nocisd, slater, pyscf_helper

# Step 1: Set up the system
nH = 4
bl = 1.5
geom = []
for i in range(nH):
    geom.append(['H', 0.0, 0.0, i*bl])

mol = gto.Mole()
mol.atom = geom
mol.unit='angstrom'
mol.basis = "631g"
mol.build()

# Step 2: Run the UHF calculation
mf = scf.UHF(mol)
mf.kernel()
mo1 = mf.stability()[0]                                                             
init = mf.make_rdm1(mo1, mf.mo_occ)                                                 
mf.kernel(init) 

# Step 3: Get attributes needed for the NOCI calculations
# I choose to orthogonalize AO, but it doesn't affect the result.
h1e, h2e, e_nuc = pyscf_helper.get_integrals(mf, ortho_ao=True)
norb, nocc, nvir, mo_coeff = pyscf_helper.get_mos(mf)
e_hf = mf.energy_tot()
nelec = mol.nelectron

# Step 4: NOCI res HF


# Step 5: Get the unitary for each NOCI


# Step 6: Run CISD and get the NOCI expansion


# Step 7: Evalaute the energy.

# myci = ci.UCISD(mf)
# e_corr, civec = myci.kernel()
# e_cisd = e_hf + e_corr 
