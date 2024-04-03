module VectorLaplacianMixedTensor_mixedBCTests

  using Gridap
  using GridapMixedViscoelasticityReactionDiffusion
  import Gridap: ∇
  using Printf
  #using LinearAlgebra

  uex(x) = VectorValue(0.05*cos(1.5*π*(x[1]+x[2])),0.05*sin(1.5*π*(x[1]-x[2])))
  σex(x) = ∇(uex)(x)
  fex(x) = -(∇⋅σex)(x)

  function extract_component(component)
    return x -> x[component]
  end


  comp1=extract_component(1)
  comp2=extract_component(2)

  # dimension-dependent 
  function extract_row2d(row)
    return x -> VectorValue(x[1,row],x[2,row])
  end

  row1=extract_row2d(1)
  row2=extract_row2d(2)

  function generate_model_unit_square(nk)
    domain =(0,1,0,1)
    n      = 2^nk
    partition = (n,n)
    model = CartesianDiscreteModel(domain, partition) |> simplexify
    model
  end

  function setup_model_labels_unit_square!(model)
    labels = get_face_labeling(model)
    add_tag!(labels,"Gamma_sig",[6,])
    add_tag!(labels,"Gamma_u",[1,2,3,4,5,7,8])
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
    Γσ = BoundaryTriangulation(model,tags = "Gamma_sig")
    Γu = BoundaryTriangulation(model,tags = "Gamma_u")
    n_Γσ = get_normal_vector(Γσ)
    n_Γu = get_normal_vector(Γu)

    Λ = SkeletonTriangulation(model)
    n_Λ = get_normal_vector(Λ)

    bdegree = 3
    dΓσ = Measure(Γσ,bdegree)
    dΓu = Measure(Γu,bdegree)

    Sh_ = TestFESpace(model,reffe_σ,dirichlet_tags="Gamma_sig",conformity=:HDiv)
    Vh_ = TestFESpace(model,reffe_u,conformity=:L2)
    #Lh_ = ConstantFESpace(model)

    Sh1 = TrialFESpace(Sh_,row1∘σex)
    Sh2 = TrialFESpace(Sh_,row2∘σex)
    Vh = TrialFESpace(Vh_)

    Y = MultiFieldFESpace([Sh_,Sh_,Vh_])
    X = MultiFieldFESpace([Sh1,Sh2,Vh])
    

    a(σ1,σ2,τ1,τ2) = ∫(σ1⋅τ1 + σ2⋅τ2)dΩ
    b(τ1,τ2,v) = ∫((comp1∘v)*(∇⋅τ1)+(comp2∘v)*(∇⋅τ2))dΩ
    F(τ1,τ2) =  ∫((τ1⋅n_Γu)*(comp1∘uex) + (τ2⋅n_Γu)*(comp2∘uex))dΓu 
    G(v) = ∫(-1.0*(fex⋅v))dΩ

    lhs((σ1,σ2,u),(τ1,τ2,v)) =  a(σ1,σ2,τ1,τ2) + b(τ1,τ2,u) + b(σ1,σ2,v)
    rhs((τ1,τ2,v)) =  F(τ1,τ2) + G(v)
    op = AffineFEOperator(lhs,rhs,X,Y) 

    σh1, σh2, uh = solve(op)

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
      setup_model_labels_unit_square!(model)
      
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
