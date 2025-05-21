module GridapMixedViscoelasticityReactionDiffusion
  using Gridap
  using Gridap.ReferenceFEs
  using Gridap.FESpaces


  include("ConstantFESpaces.jl")
  include("MixedFEMTools.jl")

  export generate_model_unit_square, setup_model_labels_unit_square!, extract_component, extract_row2d
end
