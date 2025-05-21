module ElasticityMixedTensor_pureDispl
  using Gridap
  using GridapMixedViscoelasticityReactionDiffusion
  import Gridap: ∇
  using Printf
  using LinearAlgebra

  # Material parameters
  const E = 1.0e2
  const ν = 0.49
  const λ = (E*ν)/((1+ν)*(1-2*ν))
  const μ = E/(2*(1+ν))

  calC(τ) = 2*μ*τ + λ*tr(τ)*I

  uex(x) = VectorValue(0.1*cos(π*x[1])*sin(π*x[2])+0.15/λ*x[1]^2,-0.1*sin(π*x[1])*cos(π*x[2])+0.15/λ*x[2]^2)
  σex(x) = (calC∘ε(uex))(x)
  γex(x) = 0.5*(∇(uex)(x) - transpose(∇(uex)(x)))
  fex(x) = -(∇⋅σex)(x)

  comp1=extract_component(1)
  comp2=extract_component(2)
  row1=extract_row2d(1)
  row2=extract_row2d(2)

  function solve_elasticityMixed(model; k = k, generate_output=false)

  # Reference FEs
  reffe_σ = ReferenceFE(bdm,Float64,k+1)
  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},k)
  reffe_γ = ReferenceFE(lagrangian,Float64,k)

  # Numerical integration
  degree = 5+k
  bdegree = 2+k
  Ω = Interior(model)
  dΩ = Measure(Ω,degree)

  # Boundary triangulations and outer unit normals
  Γ = BoundaryTriangulation(model)
  n_Γ = get_normal_vector(Γ)
  dΓ = Measure(Γ,bdegree)

  Sh1_ = TestFESpace(model,reffe_σ,conformity=:HDiv)
  Sh2_ = TestFESpace(model,reffe_σ,conformity=:HDiv)
  Vh_ = TestFESpace(model,reffe_u,conformity=:L2)
  Gh_ = TestFESpace(model,reffe_γ,conformity=:L2)
  Lh_ = ConstantFESpace(model)

  Sh1 = TrialFESpace(Sh1_) 
  Sh2 = TrialFESpace(Sh2_) 
  Vh = TrialFESpace(Vh_)
  Gh = TrialFESpace(Gh_)
  Lh = TrialFESpace(Lh_)

  Y = MultiFieldFESpace([Sh1_,Sh2_,Vh_,Gh_,Lh_])
  X = MultiFieldFESpace([Sh1,Sh2,Vh,Gh,Lh])

  a(σ1,σ2,τ1,τ2) = ∫(1/(2*μ)*(σ1⋅τ1 + σ2⋅τ2))dΩ -
                   ∫(λ/(2*μ*(2*μ+ 2*λ))*(comp1∘σ1+comp2∘σ2)*(comp1∘τ1+comp2∘τ2))dΩ
                   
  b(τ1,τ2,v,η) = ∫((comp1∘v)*(∇⋅τ1)+(comp2∘v)*(∇⋅τ2))dΩ + 
                 ∫(η*(comp2∘τ1 - comp1∘τ2))dΩ

  c(τ1,τ2,ζ) = ∫((comp1∘τ1+comp2∘τ2)*ζ)dΩ

  F(τ1,τ2) =  ∫((τ1⋅n_Γ)*(comp1∘uex) + (τ2⋅n_Γ)*(comp2∘uex))dΓ
  G(v) = ∫(fex⋅v)dΩ
  H(ζ) = ∫((tr∘σex)*ζ)dΩ

  lhs((σ1,σ2,u,γ,ξ),(τ1,τ2,v,η,ζ)) =  a(σ1,σ2,τ1,τ2) + b(τ1,τ2,u,γ) + b(σ1,σ2,v,η) + c(τ1,τ2,ξ) + c(σ1,σ2,ζ)
  rhs((τ1,τ2,v,η,ζ)) =  F(τ1,τ2) - G(v) + H(ζ)
  
  op = AffineFEOperator(lhs,rhs,X,Y) 

  σh1, σh2, uh, γh, _ = solve(op)

  if generate_output 
      writevtk(Ω,"convergence_AFW_natural=$(num_cells(model))",order=1,
            cellfields=["σ1"=>σh1, "σ2"=>σh2, "u"=>uh, "γ"=>γh])
  end
  
  eσ1h = (row1∘σex)-σh1
  eσ2h = (row2∘σex)-σh2
  euh  = uex-uh
  eγh  = comp2∘row1∘γex-γh
  error_σ = sqrt(sum(∫(eσ1h⋅eσ1h+eσ2h⋅eσ2h)dΩ +
                     ∫((∇⋅eσ1h)*(∇⋅eσ1h)+(∇⋅eσ2h)*(∇⋅eσ2h))dΩ))
  error_u = sqrt(sum(∫(euh⋅euh)dΩ))
  error_γ = sqrt(sum(∫(eγh*eγh)dΩ))

  error_σ,error_u, error_γ, Gridap.FESpaces.num_free_dofs(X)
  end

  function  convergence_test(; nkmax, k=0, generate_output=false)
    eσ   = Float64[]
    rσ   = Float64[]
    eu   = Float64[]
    ru   = Float64[]
    eγ   = Float64[]
    rγ   = Float64[]
    push!(ru,0.)
    push!(rσ,0.)
    push!(rγ,0.)
    nn   = Int[]
    hh   = Float64[]

    for nk in 1:nkmax
       println("******** Refinement step: $nk")
       model=generate_model_unit_square(nk) 
       error_σ, error_u, error_γ, ndofs=solve_elasticityMixed(model; k=k, generate_output=generate_output)
       push!(nn,ndofs)
       push!(hh,sqrt(2)/2^nk)
       push!(eσ,error_σ)
       push!(eu,error_u)
       push!(eγ,error_γ)

       if nk>1
         push!(rσ, log(eσ[nk]/eσ[nk-1])/log(hh[nk]/hh[nk-1]))
         push!(ru, log(eu[nk]/eu[nk-1])/log(hh[nk]/hh[nk-1]))
         push!(rγ, log(eγ[nk]/eγ[nk-1])/log(hh[nk]/hh[nk-1]))
       end
    end

    println("========================================================================")
    println("   DoF  &    h   &  e(σ)   &  r(σ)  &  e(u)   &  r(u)  & e(γ)  & r(γ)   ")
    println("========================================================================")

    for nk=1:nkmax
       @printf("%7d & %.4f & %.2e & %.3f & %.2e & %.3f & %.2e & %.3f \n",
                nn[nk], hh[nk], eσ[nk], rσ[nk], eu[nk], ru[nk], eγ[nk], rγ[nk]);
    end

    println("========================================================================")
  end
  convergence_test(;nkmax=6,k=1,generate_output=true)
end