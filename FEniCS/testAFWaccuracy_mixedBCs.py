from fenics import *
import sympy2fenics as sf

'''

Convergence history for linear elasticity with weakly imposed symmetry of stress
AFW elements of degree k >= 0
Mixed displacement/traction boundary conditions

'''

def str2exp(s):
    return sf.sympy2exp(sf.str2sympy(s))

# ********* Model coefficients and parameters ********* #

d = 2; I = Identity(d)
E     = Constant(1.0e2)
nu    = Constant(0.49)
lmbda = Constant(E*nu/((1.+nu)*(1.-2.*nu)))
mu    = Constant(E/(2.*(1.+nu)))

symgr      = lambda vec: sym(grad(vec))
skewgr     = lambda vec: grad(vec) - symgr(vec)
Cinv = lambda s: 0.5/mu * s - lmbda/(2.*mu*(d*lmbda+2.*mu))*tr(s)*Identity(d)

# ******* Exact solutions and forcing terms for error analysis ****** #
u_str = '(0.1*cos(pi*x)*sin(pi*y)+0.15*x**2/lmbda, -0.1*sin(pi*x)*cos(pi*y)+0.15*y**2/lmbda)'

k=0; nkmax = 6
hh = []; nn = []; eu = []; ru = [];
er = []; rr = []; es = []; rs = [];

rs.append(0.0); ru.append(0.0); rr.append(0.0); 

for nk in range(nkmax):
    print("....... Refinement level : nk = ", nk)

    # ******** Geometry and boundary specification **** #
    nps = pow(2,nk+1)
    mesh = UnitSquareMesh(nps,nps)
    n = FacetNormal(mesh); hh.append(mesh.hmax())
    bdry = MeshFunction("size_t", mesh, d-1)
    bdry.set_all(0)
    GammaU   = CompiledSubDomain("(near(x[0],0) || near(x[1],0)) && on_boundary")
    GammaSig = CompiledSubDomain("(near(x[0],1) || near(x[1],1)) && on_boundary")
    GU = 91; GS = 92;
    GammaU.mark(bdry,GU); GammaSig.mark(bdry,GS)
    ds = Measure("ds", subdomain_data=bdry)
    
    # ********* Finite dimensional spaces ********* #
    Bdm = FiniteElement('BDM', mesh.ufl_cell(), k+1)
    Pkv = VectorElement('DG', mesh.ufl_cell(), k)
    Pk  = FiniteElement('DG', mesh.ufl_cell(), k)
    
    Hh = FunctionSpace(mesh, MixedElement([Bdm,Bdm,Pkv,Pk]))
    nn.append(Hh.dim())
    
    # ********* test and trial functions for product space ****** #
    tau1,tau2, v, eta12 = TestFunctions(Hh)
    sig1,sig2, u, rho12 = TrialFunctions(Hh)
    
    eta=as_tensor(((0.0,eta12),
                   (-eta12,0.0)))

    rho=as_tensor(((0.0,rho12),
                   (-rho12,0.0)))

    sigma = as_tensor((sig1,sig2))
    tau = as_tensor((tau1,tau2)) 

    # ********* instantiation of exact solutions ****** #
    
    u_ex     = Expression(str2exp(u_str), lmbda=lmbda, degree=6, domain=mesh)
    rho_ex   = skewgr(u_ex)
    sigma_ex = 2.*mu*symgr(u_ex) + lmbda*div(u_ex)*I
    f        = - div(sigma_ex)

    # ********* essential BCs for sig on GammaSig and natural BCs for u on GammaU **** #
    
    sig1_ex = as_vector((sigma_ex[0,0],sigma_ex[0,1]))
    sig2_ex = as_vector((sigma_ex[1,0],sigma_ex[1,1]))
    
    sig1Sig = project(sig1_ex,Hh.sub(0).collapse())
    sig2Sig = project(sig2_ex,Hh.sub(1).collapse())
    bcSig1 = DirichletBC(Hh.sub(0),sig1Sig,bdry,GS)
    bcSig2 = DirichletBC(Hh.sub(1),sig2Sig,bdry,GS)
    
    bcs = [bcSig1,bcSig2]

    # ********* Weak forms ********* #
    
    a  = inner(Cinv(sigma),tau)*dx    
    bt = dot(u,div(tau))*dx + inner(rho,tau) *dx
    b  = dot(v,div(sigma))*dx + inner(eta,sigma) *dx 

    lhs = a + bt + b
    rhs = dot(tau*n,u_ex)*ds(GU) - dot(f,v) * dx

    Sol_h = Function(Hh)
    
    solve(lhs == rhs, Sol_h, bcs = bcs, solver_parameters={'linear_solver':"lu"})

    sig1_h,sig2_h,u_h,rho12_h = Sol_h.split()
    rho_h = as_tensor(((0,rho12_h),
                       (-rho12_h,0)))
    sigma_h = as_tensor((sig1_h,sig2_h))
    
    # ********* Computing errors ****** #

    E_s = assemble((sigma_ex-sigma_h)**2*dx \
                   +(div(sigma_ex)-div(sigma_h))**2*dx)
    E_u = assemble((u_h-u_ex)**2*dx)
    E_rho = assemble((rho_h-rho_ex)**2*dx)

    es.append(pow(E_s,0.5))
    eu.append(pow(E_u,0.5))
    er.append(pow(E_rho,0.5))
    
    if(nk>0):
        ru.append(ln(eu[nk]/eu[nk-1])/ln(hh[nk]/hh[nk-1]))
        rs.append(ln(es[nk]/es[nk-1])/ln(hh[nk]/hh[nk-1]))
        rr.append(ln(er[nk]/er[nk-1])/ln(hh[nk]/hh[nk-1]))

# ********* Generating error history ****** #
print('=======================================================================')
print('  DoF  &    h  &   e(s)   &  r(s) &   e(u)   &  r(u) &  e(r)  &  r(r)  ')
print('=======================================================================')
for nk in range(nkmax):
    print('{:6d} & {:.3f} & {:6.2e} & {:.3f} & {:6.2e} & {:.3f} & {:6.2e} & {:.3f}'.format(nn[nk], hh[nk], es[nk], rs[nk], eu[nk], ru[nk], er[nk], rr[nk]))
print('=======================================================================')
