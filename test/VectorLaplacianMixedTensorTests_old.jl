module VectorLaplacianMixedTensorTests

  using Gridap
  using GridapMixedViscoelasticityReactionDiffusion
  import Gridap: ∇
  using Printf

  uex(x) = VectorValue(20.0*x[1]*x[2]*x[2]*x[2],5.0*(x[1]*x[1]*x[1]*x[1])-5.0*(x[2]*x[2]*x[2]*x[2]))
  σex(x) = ε(uex)(x)
  fex(x) = -(∇⋅σex)(x)

  function tensorify(s1,s2)
    TensorValue((s1[1],s2[1]),(s1[2],s2[2]))
  end 

  function generate_model_unit_square(nk)
    domain =(0,1,0,1)
    n      = 2^nk
    partition = (n,n)
    model = CartesianDiscreteModel(domain, partition) |> simplexify
    model
  end

  function solve_vectorLaplacianMixed(model)
    # Reference FEs
    k = 0
    #reffe_σ = ReferenceFE(raviart_thomas,VectorValue{2,Float64},k)
    reffe_σ = ReferenceFE(raviart_thomas,Float64,k)
    reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},k)

    # Numerical integration
    degree = 2*k+1
    Ω = Interior(model)
    dΩ = Measure(Ω,degree)

    # Boundary triangulations and outer unit normals
    Γ = BoundaryTriangulation(model)
    n_Γ = get_normal_vector(Γ)

    Λ = SkeletonTriangulation(model)
    n_Λ = get_normal_vector(Λ)

    bdegree = 2*k
    dΓ = Measure(Γ,bdegree)
    dΛ = Measure(Λ,bdegree)

    h_e = CellField(get_array(∫(1)dΛ),Λ)
    h_e_Γ = CellField(get_array(∫(1)dΓ), Γ)

    Sh_ = TestFESpace(model,reffe_σ,conformity=:HDiv)
    Vh_ = TestFESpace(model,reffe_u,conformity=:L2)
    Lh_ = ConstantFESpace(model)

    Sh = TrialFESpace(Sh_)
    Vh = TrialFESpace(Vh_)
    Lh = TrialFESpace(Lh_)
    Y = MultiFieldFESpace([Sh_,Sh_,Vh_,Lh_])
    X = MultiFieldFESpace([Sh,Sh,Vh,Lh])

    #σ = TensorValue(σ1,σ2)
    #τ = TensorValue(τ1,τ2)


    a((σ1,σ2),(τ1,τ2)) = ∫(tensorify(σ1,σ2)⊙tensorify(τ1,τ2))dΩ
    b((τ1,τ2),v) = ∫(v⋅(∇⋅tensorify(τ1,τ2)))dΩ

    c((τ1,τ2),ψ) = ∫(tr(tensorify(τ1,τ2))*ψ)dΩ

    F((τ1,τ2)) = ∫((tensorify(τ1,τ2)⋅n_Γ)⋅uex)dΓ
    G(v) = ∫(-fex⋅v)dΩ
    H(ψ) = ∫(tr(σex)*ψ)dΩ
    
    lhs((σ1,σ2,u,φ),(τ1,τ2,v,ψ)) =  a((σ1,σ2),(τ1,τ2)) + b((τ1,τ2),u) + b((σ1,σ2),v) + c((τ1,τ2),φ) + c((σ1,σ2),ψ) 

    rhs((τ1,τ2,v,ψ)) = F((τ1,τ2)) + G(v) + H(ψ)

    op = AffineFEOperator(lhs,rhs,X,Y) 

    σh1, σh2, uh, φh = solve(op)

    σh = tensorify(σh1,σh2)

    error_σ = sqrt(sum(∫((σex-σh)⊙(σex-σh) + (∇⋅(σex-σh))⋅(∇⋅(σex-σh)))dΩ)) 
    error_u = sqrt(sum(∫(uex-uh)⋅(uex - uh)*dΩ))

    error_σ, error_u, Gridap.FESpaces.num_free_dofs(X)
  end

  eσ   = Float64[]
  rσ   = Float64[]
  eu   = Float64[]
  ru   = Float64[]

  push!(ru,0.)
  push!(rσ,0.)

  m=generate_model_unit_square(1)
  solve_vectorLaplacianMixed(m)
  nn   = Int[]
  hh   = Float64[]

  nkmax = 6
  for nk in 1:nkmax
      println("******** Refinement step: $nk")
      model=generate_model_unit_square(nk) # Discrete model
      println("numcells of model: $(num_cells(model))")
      error_σ, error_u, ndofs=solve_vectorLaplacianMixed(model)
      push!(nn,ndofs)
      push!(hh,sqrt(2)/2^nk)
      push!(eσ,error_σ)
      push!(eu,error_u)

      if nk>1
        push!(ru, log(eσ[nk]/eσ[nk-1])/log(hh[nk]/hh[nk-1]))
        push!(ru, log(eu[nk]/eu[nk-1])/log(hh[nk]/hh[nk-1]))
      end
  end

  println("==========================================================")
  println("   DoF  &    h   &  e(σ)   &  r(σ)  &  e(u)   &  r(u)     ")
  println("==========================================================")

  for nk=1:nkmax
      @printf("%7d & %.4f & %.2e & %.3f & %.2e & %.3f \n",
               nn[nk], hh[nk], eσ[nk], rσ[nk], eu[nk], ru[nk]);
  end

  println("==========================================================")

end
