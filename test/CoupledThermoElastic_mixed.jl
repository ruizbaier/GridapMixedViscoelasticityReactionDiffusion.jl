module CoupledThermoElastic_mixed
  #-----------------------------------------------------------------------------
  # Libraries
  #-----------------------------------------------------------------------------
  using Gridap
  import Gridap: ∇
  using Printf
  using Gridap.TensorValues

  function extract_component(component)
    return x -> x[component]
  end

# dimension-dependent 
  function extract_row2d(row)
    return x -> VectorValue(x[1,row],x[2,row])
  end

  comp1=extract_component(1)
  comp2=extract_component(2)
  row1=extract_row2d(1)
  row2=extract_row2d(2)

  #-----------------------------------------------------------------------------
  # Mesh and labels
  #-----------------------------------------------------------------------------

  function generate_model_unit_square(nk)
    domain =(0,1,0,1)
    n      = 2^nk
    partition = (n,n)
    model = CartesianDiscreteModel(domain, partition) |> simplexify
    model
  end

  function setup_model_labels_unit_square!(model)
    labels = get_face_labeling(model)
    add_tag!(labels,"Gamma_N",[6,])
    add_tag!(labels,"Gamma_D",[1,2,3,4,5,7,8])
  end


  #------------------------------------------------------------------------------
  #Exact data and parameters
  #------------------------------------------------------------------------------

  calC(τ) = 2*μ*τ + λ*tr(τ)*one(τ)

  uex(x) = VectorValue(0.1*cos(π*x[1])*sin(π*x[2])+0.15/λ*x[1]^2,
                      -0.1*sin(π*x[1])*cos(π*x[2])+0.15/λ*x[2]^2)

  φex(x) = 0.5 + 0.5*cos(x[1]*x[2])
  # Material parameters
  const E = 1.0e2
  const ν = 0.4999
  const λ = (E*ν)/((1+ν)*(1-2*ν))
  const μ = E/(2*(1+ν))

  K = TensorValue(0.1, 0.0, 0.0, 0.1)
  const α = 0.1
  const β = 1.0
  const γ = 1/(2λ + 3μ)

  σex(x) = (calC∘ε(uex))(x) +α*γ*φex(x)*one(SymTensorValue{2,Float64})
  γex(x) = 0.5*(∇(uex)(x) - transpose(∇(uex)(x)))
  fex(x) = -(∇⋅σex)(x)

  ∇φex(x) = ∇(φex)(x)
  θex(x)  = K⋅∇φex(x)
  fexH(x)   = -(∇⋅θex)(x) + φex(x) + (β*(uex(x)⋅(inv(K)⋅θex(x))))

  #------------------------------------------------------------------------------
  # Solver
  #------------------------------------------------------------------------------

  function solve_coupled_elasticity_heat_mixed(model; k = k, generate_output=false)

  # Reference FEs
  reffe_θ = ReferenceFE(bdm,Float64,k+1)
  reffe_φ = ReferenceFE(lagrangian,Float64,k)

  reffe_σ = ReferenceFE(bdm,Float64,k+1)
  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},k)
  reffe_γ = ReferenceFE(lagrangian,Float64,k)

  # Numerical integration
  degree = 5+k
  Ω = Interior(model)
  dΩ = Measure(Ω,degree)

  # Boundary triangulations and outer unit normals
  ΓN = BoundaryTriangulation(model,tags = "Gamma_N")
  ΓD = BoundaryTriangulation(model,tags = "Gamma_D")
  n_ΓD = get_normal_vector(ΓD)
  dΓD = Measure(ΓD,degree)

  ShH_ = TestFESpace(model,reffe_θ,dirichlet_tags="Gamma_N",conformity=:HDiv)
  VhH_ = TestFESpace(model,reffe_φ,conformity=:L2)

  ShH = TrialFESpace(ShH_,θex)
  VhH = TrialFESpace(VhH_)

  YH = MultiFieldFESpace([ShH_,VhH_])
  XH = MultiFieldFESpace([ShH,VhH])

  ShE_ = TestFESpace(model,reffe_σ,dirichlet_tags="Gamma_N",conformity=:HDiv)
  VhE_ = TestFESpace(model,reffe_u,conformity=:L2)
  GhE_ = TestFESpace(model,reffe_γ,conformity=:L2)

  ShE1 = TrialFESpace(ShE_,row1∘σex)
  ShE2 = TrialFESpace(ShE_,row2∘σex)
  VhE = TrialFESpace(VhE_)
  GhE = TrialFESpace(GhE_)

  YE = MultiFieldFESpace([ShE_,ShE_,VhE_,GhE_])
  XE = MultiFieldFESpace([ShE1,ShE2,VhE,GhE])

  #---------------------------------------------------------------------
  # Heat bilinear forms and functionals
  #---------------------------------------------------------------------

  aH(ζ,ξ) = ∫((inv(K)⋅ζ)⋅ξ)dΩ
  bH(ξ,ψ) = ∫(ψ*(∇⋅ξ))dΩ
  cH(ψ,χ) = ∫(ψ*χ)dΩ

  FH(ξ) =  ∫((ξ⋅n_ΓD)*φex)dΓD
  GH(ψ) =  ∫(fexH*ψ)dΩ

  #the LHS of Heat depends on FP argument
  rhsH((ξ,ψ)) =  FH(ξ) - GH(ψ) 

  #----------------------------------------------------------------------
  # Elasticity bilinear forms and functionals
  #----------------------------------------------------------------------


  aE(σ1,σ2,τ1,τ2) = ∫(1/(2*μ)*(σ1⋅τ1 + σ2⋅τ2))dΩ -
                   ∫(λ/(2*μ*(2*μ+ 2*λ))*(comp1∘σ1+comp2∘σ2)*(comp1∘τ1+comp2∘τ2))dΩ
 
  bE(τ1,τ2,v,η) = ∫((comp1∘v)*(∇⋅τ1)+(comp2∘v)*(∇⋅τ2) + η*(comp2∘τ1-comp1∘τ2))dΩ

  FE(τ1,τ2) =  ∫((τ1⋅n_ΓD)*(comp1∘uex) + (τ2⋅n_ΓD)*(comp2∘uex))dΓD 
  GE(v) = ∫(fex⋅v)dΩ

  lhsE((σ1,σ2,u,γ),(τ1,τ2,v,η)) =  aE(σ1,σ2,τ1,τ2) + bE(τ1,τ2,u,γ) + bE(σ1,σ2,v,η) 


  #----------------------------------------------------------------------
  # Fixed point iteration
  #----------------------------------------------------------------------

  tol = 1e-6
  max_iter = 15
  err = 1.0
  it = 0
  
  #initial uh
  uh = interpolate_everywhere(x -> VectorValue(0.0, 0.0), VhE) 
  θh = zero(ShH)
  φh = zero(VhH)
  σh1 = zero(ShE1)
  σh2 = zero(ShE2)
  γh  = zero(GhE)

  while err > tol && it < max_iter

    it += 1
    println("*** Fixed point iteration $it")
    ########################### Heat
    # Assemble Heat LHS 
    lhsH((θ,φ),(ξ,ψ)) =  aH(θ,ξ) + bH(ξ,φ) + bH(θ,ψ) - cH(φ,ψ) - ∫((uh⋅(inv(K)⋅θ))*ψ)dΩ

    println("*** Assembling Diffusive LHS, FPI: $it")
    opH = AffineFEOperator(lhsH,rhsH,XH,YH)

    println("*** Solving Diffusive model")
    θh, φh = solve(opH)

    #@show typeof(φh)

    ######################### Elasticity
    # Assemble Elasticity RHS
    rhsE((τ1,τ2,v,η)) = FE(τ1,τ2) - GE(v)  + ∫(α*γ*(φh*(comp1∘τ1 + comp2∘τ2)))dΩ

    println("*** Assembling Elasticity RHS, FPI: $it")
    opE = AffineFEOperator(lhsE,rhsE,XE,YE)

    println("*** Solving Elasticity model")
    σh1, σh2, uh_new, γh = solve(opE)

    ######################## Tolerance
    println("*** Calculating tolerance")
    diff = uh_new - uh
    err = sqrt(sum(∫(diff⋅diff)dΩ))

    println(">> Error: $err")
    uh = uh_new

  end

  #------------------------------------------------------------------------
  #Errors
  #------------------------------------------------------------------------

  println("******** Calculating error")
  eθh  = θex-θh
  eφh  = φex-φh
  eσ1h = (row1∘σex)-σh1
  eσ2h = (row2∘σex)-σh2
  euh  = uex-uh
  eγh  = comp2∘row1∘γex-γh

  error_φ = sqrt(sum(∫(eφh⋅eφh)dΩ))
  error_θ = sqrt(sum(∫(eθh⋅eθh)dΩ + ∫((∇⋅eθh)*(∇⋅eθh))dΩ))


  error_σ = sqrt(sum(∫(eσ1h⋅eσ1h+eσ2h⋅eσ2h)dΩ +
                     ∫((∇⋅eσ1h)*(∇⋅eσ1h)+(∇⋅eσ2h)*(∇⋅eσ2h))dΩ))
  error_u = sqrt(sum(∫(euh⋅euh)dΩ))
  error_γ = sqrt(sum(∫(eγh*eγh)dΩ))

  error_θ, error_φ, error_σ, error_u, error_γ, Gridap.FESpaces.num_free_dofs(XE) + Gridap.FESpaces.num_free_dofs(XH), it
  
  end

  #--------------------------------------------------------------------------------
  #
  #--------------------------------------------------------------------------------

  function  convergence_test(; nkmax, k=0, generate_output=false)
    eθ   = Float64[]
    rθ   = Float64[]
    eφ   = Float64[]
    rφ   = Float64[]
    eσ   = Float64[]
    rσ   = Float64[]
    eu   = Float64[]
    ru   = Float64[]
    eγ   = Float64[]
    rγ   = Float64[]
    niter = Int[]

    push!(rθ,0.)
    push!(rφ,0.)
    push!(ru,0.)
    push!(rσ,0.)
    push!(rγ,0.)

    nn   = Int[]
    hh   = Float64[]

    for nk in 1:nkmax
       println("**************** Refinement step: $nk ****************" )
       model=generate_model_unit_square(nk) # Discrete model
       setup_model_labels_unit_square!(model)
      
       error_θ, error_φ, error_σ, error_u, error_γ, ndofs, n_it =solve_coupled_elasticity_heat_mixed(model; k=k, generate_output=generate_output)
       push!(nn,ndofs)
       push!(hh,sqrt(2)/2^nk)
       push!(eθ,error_θ)
       push!(eφ,error_φ)
       push!(eσ,error_σ)
       push!(eu,error_u)
       push!(eγ,error_γ)
       push!(niter, n_it)

       if nk>1
         push!(rθ, log(eθ[nk]/eθ[nk-1])/log(hh[nk]/hh[nk-1]))
         push!(rφ, log(eφ[nk]/eφ[nk-1])/log(hh[nk]/hh[nk-1]))
         push!(rσ, log(eσ[nk]/eσ[nk-1])/log(hh[nk]/hh[nk-1]))
         push!(ru, log(eu[nk]/eu[nk-1])/log(hh[nk]/hh[nk-1]))
         push!(rγ, log(eγ[nk]/eγ[nk-1])/log(hh[nk]/hh[nk-1]))
       end
    end

println("=============================================================================================================================")
println("   DoF  &    h   &  e(σ)   &  r(σ)  &  e(u)   &  r(u)  &  e(γ)  & r(γ)  &  e(θ)   &  r(θ)  &  e(φ)   &  r(φ)  &  #it")
println("=============================================================================================================================")

for nk = 1:nkmax
  @printf("%7d & %.4f & %.2e & %.3f & %.2e & %.3f & %.2e & %.3f & %.2e & %.3f & %.2e & %.3f & %4d\n",
          nn[nk], hh[nk],
          eσ[nk], rσ[nk],
          eu[nk], ru[nk],
          eγ[nk], rγ[nk],
          eθ[nk], rθ[nk],
          eφ[nk], rφ[nk],
          niter[nk])
end

println("=============================================================================================================================")

  end
  convergence_test(;nkmax=7,k=0,generate_output=true)


end