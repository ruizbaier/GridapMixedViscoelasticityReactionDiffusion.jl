module HeatMixedVectorial_mixedBCTest
  #-----------------------------------------------------------------------------
  
  # Libraries

  #-----------------------------------------------------------------------------
  using Gridap
  import Gridap: ∇
  using Printf
  using LinearAlgebra
  using Gridap.TensorValues

  #push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
  #using GridapMixedViscoelasticityReactionDiffusion

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
    add_tag!(labels,"Gamma_th",[6,])
    add_tag!(labels,"Gamma_phi",[1,2,3,4,5,7,8])
  end

  #------------------------------------------------------------------------------

  #Exact data

  #------------------------------------------------------------------------------

  K = TensorValue(0.1, 0.0, 0.0, 0.1)
  φex(x) = 0.5 + 0.5*cos(x[1]*x[2])
  ∇φex(x) = ∇(φex)(x)
  θex(x)  = K⋅∇φex(x)
  fex(x)   = -(∇⋅θex)(x) + φex(x)

  #------------------------------------------------------------------------------

  # Solver

  #------------------------------------------------------------------------------

  function solve_heat_mixed(model; k = k, generate_output=false)

  # Reference FEs
  reffe_θ = ReferenceFE(bdm,Float64,k+1)
  reffe_φ = ReferenceFE(lagrangian,Float64,k)

  # Numerical integration
  degree = 5+k
  Ω = Interior(model)
  dΩ = Measure(Ω,degree)

  # Boundary triangulations and outer unit normals
  Γθ = BoundaryTriangulation(model,tags = "Gamma_th")
  Γφ = BoundaryTriangulation(model,tags = "Gamma_phi")
  n_Γφ = get_normal_vector(Γφ)
  dΓφ = Measure(Γφ,degree)

  Sh_ = TestFESpace(model,reffe_θ,dirichlet_tags="Gamma_th",conformity=:HDiv)
  Vh_ = TestFESpace(model,reffe_φ,conformity=:L2)

  Sh = TrialFESpace(Sh_,θex)
  Vh = TrialFESpace(Vh_)

  Y = MultiFieldFESpace([Sh_,Vh_])
  X = MultiFieldFESpace([Sh,Vh])

  a(ζ,ξ) = ∫((inv(K)⋅ζ)⋅ξ)dΩ
  b(ξ,ψ) = ∫(ψ*(∇⋅ξ))dΩ
  c(ψ,χ) = ∫(ψ*χ)dΩ

  F(ξ) =  ∫((ξ⋅n_Γφ)*φex)dΓφ
  G(ψ) =  ∫(fex*ψ)dΩ

  lhs((θ,φ),(ξ,ψ)) =  a(θ,ξ) + b(ξ,φ) + b(θ,ψ) - c(φ,ψ) 
  rhs((ξ,ψ)) =  F(ξ) - G(ψ) 

  op = AffineFEOperator(lhs,rhs,X,Y) 

  #println("******** Solving linear system")

  θh,φh = solve(op)

  #------------------------------------------------------------------------------------

  #Paraview export

  #------------------------------------------------------------------------------------

  #println("******** Generating .vtu")

  if generate_output
    vtk_dir = joinpath(@__DIR__, "paraview-data")
      writevtk(Ω,
               joinpath(vtk_dir,"convergence_AFW=$(num_cells(model))"),
               order=1,
               cellfields=["θ"=>θh, "φ"=>φh])
      writevtk(model,joinpath(vtk_dir,"model"))
  end

  #------------------------------------------------------------------------------------

  #Errors

  #------------------------------------------------------------------------------------
  
  #println("******** Calculating error")
  eθh  = θex-θh
  eφh  = φex-φh

  error_φ = sqrt(sum(∫(eφh⋅eφh)dΩ))
  error_θ = sqrt(sum(∫(eθh⋅eθh)dΩ + ∫((∇⋅eθh)*(∇⋅eθh))dΩ))

  error_θ, error_φ, Gridap.FESpaces.num_free_dofs(X)
  end
  

  function  convergence_test(; nkmax, k=0, generate_output=false)
    eθ   = Float64[]
    rθ   = Float64[]
    eφ   = Float64[]
    rφ   = Float64[]
    push!(rθ,0.)
    push!(rφ,0.)
    nn   = Int[]
    hh   = Float64[]

    for nk in 1:nkmax
       println("**************** Refinement step: $nk ****************" )
       model=generate_model_unit_square(nk) # Discrete model
       setup_model_labels_unit_square!(model)
      
       error_θ, error_φ, ndofs=solve_heat_mixed(model; k=k, generate_output=generate_output)
       push!(nn,ndofs)
       push!(hh,sqrt(2)/2^nk)
       push!(eθ,error_θ)
       push!(eφ,error_φ)

       if nk>1
         push!(rθ, log(eθ[nk]/eθ[nk-1])/log(hh[nk]/hh[nk-1]))
         push!(rφ, log(eφ[nk]/eφ[nk-1])/log(hh[nk]/hh[nk-1]))
       end
    end

    println("==============================================================")
    println("   DoF  &    h   &  e(θ)   &  r(θ)  &  e(φ)   &  r(φ)          ")
    println("==============================================================")

    for nk = 1:nkmax
      @printf("%7d & %.4f & %.2e & %.3f & %.2e & %.3f \n",
              nn[nk], hh[nk], eθ[nk], rθ[nk], eφ[nk], rφ[nk]);
    end
    
    println("==============================================================")

  end
  convergence_test(;nkmax=7,k=0,generate_output=true)
end