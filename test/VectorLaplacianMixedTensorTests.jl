module VectorLaplacianMixedTensorTests

  using Gridap
  using GridapMixedViscoelasticityReactionDiffusion
  import Gridap: ∇
  using Printf
  using LinearAlgebra

  uex(x) = VectorValue(0.05*cos(1.5*π*(x[1]+x[2])),0.05*sin(1.5*π*(x[1]-x[2])))
  σex(x) = ∇(uex)(x)
  fex(x) = -(∇⋅σex)(x)

  function extract_component(component)
    return x -> x[component]
  end

  function extract_row(row)
    return x -> VectorValue(x[1,row],x[2,row])
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
    reffe_σ = ReferenceFE(raviart_thomas,Float64,k)
    reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},k)

    # Numerical integration
    degree = 5
    Ω = Interior(model)
    dΩ = Measure(Ω,degree)

    # Boundary triangulations and outer unit normals
    Γ = BoundaryTriangulation(model)
    n_Γ = get_normal_vector(Γ)

    Λ = SkeletonTriangulation(model)
    n_Λ = get_normal_vector(Λ)

    bdegree = 3
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

    comp1=extract_component(1)
    comp2=extract_component(2)

    a(σ1,σ2,τ1,τ2) = ∫(σ1⋅τ1 + σ2⋅τ2)dΩ
    b(τ1,τ2,v) = ∫((comp1∘v)*(∇⋅τ1)+(comp2∘v)*(∇⋅τ2))dΩ
    c(τ1,τ2,ζ) = ∫((comp1∘τ1+comp2∘τ2)*ζ)dΩ
    F(τ1,τ2) =  ∫((τ1⋅n_Γ)*(comp1∘uex) + (τ2⋅n_Γ)*(comp2∘uex))dΓ
    
    #∫((tensorify(τ1,τ2)⋅(c1∘n_Γ))⋅uex)dΓ
    G(v) = ∫(-1.0*(fex⋅v))dΩ
    H(ζ) = ∫((tr∘σex)*ζ)dΩ
    lhs((σ1,σ2,u,ξ),(τ1,τ2,v,ζ)) =  a(σ1,σ2,τ1,τ2) + c(τ1,τ2,ξ) + c(σ1,σ2,ζ) + b(τ1,τ2,u) + b(σ1,σ2,v)
    rhs((τ1,τ2,v,ζ)) =  F(τ1,τ2) + G(v) + H(ζ)
    op = AffineFEOperator(lhs,rhs,X,Y) 
    #println(eigvals(Array(op.op.matrix)))
    σh1, σh2, uh, ξh = solve(op)

    row1=extract_row(1)
    row2=extract_row(2)
    
    error_σ = sqrt(sum(
                ∫( ((row1∘σex)-σh1)⋅((row1∘σex)-σh1) + ((row2∘σex)-σh2)⋅((row2∘σex)-σh2))dΩ
                + ∫( (∇⋅(row1∘σex)-∇⋅σh1)*(∇⋅(row1∘σex)-∇⋅σh1) + (∇⋅(row2∘σex)-∇⋅σh2)*(∇⋅(row2∘σex)-∇⋅σh2))dΩ))
    error_u = sqrt(sum(∫((uex-uh)⋅(uex-uh))dΩ))
    error_σ,error_u, Gridap.FESpaces.num_free_dofs(X)
  end

  eσ   = Float64[]
  rσ   = Float64[]
  eu   = Float64[]
  ru   = Float64[]

  push!(ru,0.)
  push!(rσ,0.)

  nn   = Int[]
  hh   = Float64[]

  nkmax = 6
  for nk in 1:nkmax
      println("******** Refinement step: $nk")
      model=generate_model_unit_square(nk) # Discrete model
      #println("numcells of model: $(num_cells(model))")
      error_σ, error_u, ndofs=solve_vectorLaplacianMixed(model)
      push!(nn,ndofs)
      push!(hh,sqrt(2)/2^nk)
      push!(eσ,error_σ)
      push!(eu,error_u)

      if nk>1
        push!(rσ, log(eσ[nk]/eσ[nk-1])/log(hh[nk]/hh[nk-1]))
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
