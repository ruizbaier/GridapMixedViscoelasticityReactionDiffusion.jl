from fenics import *
import sympy2fenics as sf

'''
Convergence history for mixed vector laplacian in terms of stress and displacement

RT elements (no need to impose symmetry)

Pure displacement boundary conditions, therefore the trace of sigma 
   is fixed through a real Lagrange multiplier
'''

def str2exp(s):
    return sf.sympy2exp(sf.str2sympy(s))

# ********* Model coefficients and parameters ********* #

d = 2; I = Identity(d)
lmbda = Constant(1.)

# ******* Exact solutions and forcing terms for error analysis ****** #
u_str = '(0.05*cos(1.5*pi*(x+y)),0.05*sin(1.5*pi*(x-y)))'

k=0; nkmax = 5
hh = []; nn = []; eu = []; ru = [];
er = []; rr = []; es = []; rs = [];

rs.append(0.0); ru.append(0.0); rr.append(0.0); 

for nk in range(nkmax):
    print("....... Refinement level : nk = ", nk)
    nps = pow(2,nk+1)
    mesh = UnitSquareMesh(nps,nps)
    n = FacetNormal(mesh); hh.append(mesh.hmax())

    # ********* Finite dimensional spaces ********* #
    Pkv = VectorElement('DG', mesh.ufl_cell(), k)
    RT  = FiniteElement('RT', mesh.ufl_cell(), k+1)
    R0  = FiniteElement('R', mesh.ufl_cell(), 0)
    
    Hh = FunctionSpace(mesh, MixedElement([RT,RT,Pkv,R0])) 
    nn.append(Hh.dim())
    
    # ********* test and trial functions for product space ****** #
    tau1, tau2, v, zeta = TestFunctions(Hh)
    sig1, sig2, u, xi = TrialFunctions(Hh)

    sigma = as_tensor((sig1,sig2))
    tau   = as_tensor((tau1,tau2))

    # ********* instantiation of exact solutions ****** #
    
    u_ex     = Expression(str2exp(u_str), degree=6, domain=mesh)
    sigma_ex = grad(u_ex)
    uD       = u_ex
    f        = - div(sigma_ex)

    # ********* Weak forms ********* #
    
    aa  = inner(sigma,tau)*dx    
    bbt = dot(u,div(tau))*dx 
    bb  = dot(v,div(sigma))*dx 

    Btilde = aa + bbt + bb + tr(sigma) * zeta * dx + tr(tau) * xi * dx

    Ftilde = dot(tau*n,uD)*ds - dot(f,v) * dx + tr(sigma_ex) * zeta * dx 

    Sol_h = Function(Hh)
    
    solve(Btilde == Ftilde, Sol_h)

    sig1_h,sig2_h,u_h,xi_h = Sol_h.split()

    sigma_h = as_tensor((sig1_h,sig2_h))
    
    
    # ********* Computing errors ****** #

    E_s = assemble((sigma_ex-sigma_h)**2*dx \
                   +(div(sigma_ex)-div(sigma_h))**2*dx)

    es.append(pow(E_s,0.5))
    eu.append(errornorm(u_ex,u_h,'L2'))
    
    if(nk>0):
        ru.append(ln(eu[nk]/eu[nk-1])/ln(hh[nk]/hh[nk-1]))
        rs.append(ln(es[nk]/es[nk-1])/ln(hh[nk]/hh[nk-1]))

# ********* Generating error history ****** #
print('===============================================================')
print('  DoF  &   e(s)   &  r(s) &   e(u)   &  r(u)  ')
print('===============================================================')
for nk in range(nkmax):
    print('{:6d} & {:6.2e} & {:.3f} & {:6.2e} & {:.3f}'.format(nn[nk], es[nk], rs[nk], eu[nk], ru[nk]))
print('===============================================================')
