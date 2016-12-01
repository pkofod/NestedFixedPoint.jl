module NestedFixedPoint

using StocasEstimatorsBase, Optim
import StocasEstimatorsBase: fit, inner_iterations, outer_iterations
export fit_nfxp, fit, NFXP

include("types.jl")
include("logit.jl")
include("fit.jl")
end # module
