#-----------------------------------------------------------------------------
  # Libraries
  #-----------------------------------------------------------------------------
  using Gridap
  import Gridap: ∇
  using Printf
  using Gridap.TensorValues


  #L2 projection on ΓC with discontinuous Lagrange polinomials
  function project_on_contact_boundary(τ,Λh,dΛh)
    a(u,v) = ∫(u⋅v)dΛh
    l(v) = ∫(v⋅τ)dΛh
    op = AffineFEOperator(a,l,Λh,Λh)
    Πτh = solve(op)
    Πτh
  end

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

  AssembleTensor(σ1,σ2)       = TensorValues.tensor_from_rows(σ1,σ2)
  AssembleVector(divσ1,divσ2) = VectorValue(divσ1,divσ2)
  AssembleTrace(σ1,σ2)        = comp1∘σ1+comp2∘σ2

  ComputeNormalStress(σ,n) = n⋅σ⋅n
  ComputeTangentialStress(σ,n) = σ⋅n - ComputeNormalStress(σ,n)⋅n



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
    add_tag!(labels,"Gamma_N",[4,8])
    add_tag!(labels,"Gamma_D",[5,7,6])
    add_tag!(labels,"Gamma_C",[1,2,3])
  end

  k  = 0
  nk = 2

  model=generate_model_unit_square(nk) # Discrete model
  setup_model_labels_unit_square!(model)


  #------------------------------------------------------------------------------
  #Exact data and parameters
  #------------------------------------------------------------------------------

  #T(n) = one(n⊗n) - n⊗n

  calC(τ) = 2*μ*τ + λ*tr(τ)*one(τ)

  uex(x) = VectorValue(0.1*cos(π*x[1])*sin(π*x[2])+0.15/λ*x[1]^2,
                      -0.1*sin(π*x[1])*cos(π*x[2])+0.15/λ*x[2]^2)

  # Material parameters
  const E = 1.0e2
  const ν = 0.4999
  const λ = (E*ν)/((1+ν)*(1-2*ν))
  const μ = E/(2*(1+ν))

  σex(x) = (calC∘ε(uex))(x)
  γex(x) = 0.5*(∇(uex)(x) - transpose(∇(uex)(x)))
  fex(x) = -(∇⋅σex)(x)

  #------------------------------------------------------------------------------
  # Solver
  #------------------------------------------------------------------------------

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
  ΓC = BoundaryTriangulation(model,tags = "Gamma_C")
  n_ΓC = get_normal_vector(ΓC)
  n_ΓD = get_normal_vector(ΓD)
  dΓC = Measure(ΓC,degree)
  dΓD = Measure(ΓD,degree)

  Sh_ = TestFESpace(model,reffe_σ,dirichlet_tags="Gamma_N",conformity=:HDiv)
  Vh_ = TestFESpace(model,reffe_u,conformity=:L2)
  Gh_ = TestFESpace(model,reffe_γ,conformity=:L2)

  Sh1 = TrialFESpace(Sh_,row1∘σex)
  Sh2 = TrialFESpace(Sh_,row2∘σex)
  Vh = TrialFESpace(Vh_)
  Gh = TrialFESpace(Gh_)

  Y = MultiFieldFESpace([Sh_,Sh_,Vh_,Gh_])
  X = MultiFieldFESpace([Sh1,Sh2,Vh,Gh])

  #----------------------------------------------------------------------
  # Elasticity bilinear forms and functionals
  #----------------------------------------------------------------------

  #a_sf1(σ,τ) = ∫(1/(2*μ)*(σ⋅τ))dΩ - ∫(λ/(2*μ*(2*μ+ 2*λ))*σ*tr(τ) )dΩ
  #b_sf(τ,v) = ∫
  #a((σ1,σ2),(τ1,τ2)) = a_sf(to_σ∘(σ1,σ2),to_σ∘(τ1,τ2))

  a(σ1,σ2,τ1,τ2) = ∫(1/(2*μ)*(σ1⋅τ1 + σ2⋅τ2))dΩ -
                   ∫(λ/(2*μ*(2*μ+ 2*λ))*(comp1∘σ1+comp2∘σ2)*(comp1∘τ1+comp2∘τ2))dΩ
 
  b(τ1,τ2,v,η) = ∫((comp1∘v)*(∇⋅τ1)+(comp2∘v)*(∇⋅τ2) + η*(comp2∘τ1-comp1∘τ2))dΩ

  F(τ1,τ2) =  ∫((τ1⋅n_ΓD)*(comp1∘uex) + (τ2⋅n_ΓD)*(comp2∘uex))dΓD 
  G(v) = ∫(fex⋅v)dΩ

  rhs((τ1,τ2,v,η)) =  F(τ1,τ2) - G(v)  


  #----------------------------------------------------------------------
  # Uzawa
  #----------------------------------------------------------------------

  println("***** Entering Uzawa")

  GLOBAL_TOL = 1e-10
  MAX_ITER = 10
  it = 0
  
  # Reference FEs for multipliers
  reffe_ρ = ReferenceFE(lagrangian,Float64,k)
  Λh = FESpace(ΓC, reffe_ρ; conformity=:L2)
  #dΛh = Measure(Λh,degree)

  ρ = zero(Λh)
  err = 1.0
  uzawa_step = 10.0
  γ0 = 10.0
  h_F = CellField(get_array(∫(1)dΓC), ΓC)
  println("h_F = $h_F")


  a_uzawa_tang((σ1,σ2),(τ1,τ2)) = begin
    σ = AssembleTensor∘(σ1,σ2)
    τ = AssembleTensor∘(τ1,τ2)
    σT = ComputeTangentialStress∘(σ,n_ΓC)
    τT = ComputeTangentialStress∘(τ,n_ΓC)
    F = ((γ0./h_F)*(σT⋅τT))
    ∫(F)dΓC
  end



    it = 1

    println("*** Uzawa step $it")
    a_uzawa_normal((σ1,σ2),(τ1,τ2)) = begin
     σ = AssembleTensor∘(σ1,σ2)
     τ = AssembleTensor∘(τ1,τ2)
     σN = ComputeNormalStress∘(σ,n_ΓC)
     τN = ComputeNormalStress∘(σ,n_ΓC)
     ∫((γ0./h_F)*(ρ⋅τN))dΓC
    end
    
    lhs((σ1,σ2,u,γ),(τ1,τ2,v,η)) =  a(σ1,σ2,τ1,τ2) + b(τ1,τ2,u,γ) + b(σ1,σ2,v,η)   + a_uzawa_tang((σ1,σ2),(τ1,τ2)) + a_uzawa_normal((σ1,σ2),(τ1,τ2))
    println("*** Assembling")
    op = AffineFEOperator(lhs,rhs,X,Y) 

    println("*** Solving")
    σh1, σh2, uh, γh = solve(op)
    
    σh1_projected = project_on_contact_boundary(σh1,Λh,dΓC)

    σh = AssembleTensor∘(σh1,σh2)
    σhN = GET_NORMAL∘(σh,n_ΓC)
    println(typeof(σhN)) 

    σhN_projected = project_on_contact_boundary∘(σhN,model,dΓC,k)
    ρ_new = ρ + uzawa_step*σhN_projected

    println("ρ = $ρ_new")

    eρ = (ρ_new-ρ)
    err = sqrt(sum(∫((eρ)⋅(eρ))dΓC))
    println("error: $err")

    ρ = interpolate_everywhere(max.(ρ_new,0.0), Λh)
    println("ρ interpolated = $ρ")

