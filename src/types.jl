type TraceNFXP{Tv, Tm} <: AbstractTrace
    Θ::Tm
    norm_ΔΘ::Tv
    ll::Tv
    Δll::Tv
    norm_g::Tv
end

TraceNFXP(K, n_var) = TraceNFXP(zeros(K, n_var),
                                zeros(K),
                                zeros(K),
                                zeros(K),
                                zeros(K))

type ConvergenceNFXP <: ConvergenceInfo
    # Outer loop
    flag::Bool
    outer::Int64
    # Inner loop
    iter::Vector{Int64}
    norm_sol::Float64
end
ConvergenceNFXP() = ConvergenceNFXP(false, 0, [0; 0], 0.)

outer_iterations(conv::ConvergenceNFXP) = conv.outer
inner_iterations(conv::ConvergenceNFXP) = conv.iter
contraction_iterations(conv::ConvergenceNFXP) = inner_iterations(conv)[1]
newton_iterations(conv::ConvergenceNFXP) = inner_iterations(conv)[2]


type NFXP{Tu<:AbstractUtility, Tm<:AbstractMatrix,
          S<:Stocas.AbstractState, Tv<:Stocas.AbstractValueFunction} <: EstimationMethod
    U::Tu
    X::S
    D::Data
    P::Vector{Vector{Float64}}
    logP::Vector{Vector{Float64}}
    Fᵁ::Tm
    V::Tv
    v::Vector{Vector{Float64}}
    dVu::Matrix{Float64}
    verbose::Bool
    K
end

NFXP(U, X, D; verbose = false, K = 50) = NFXP(U, X, D, [zeros(X.nX) for p = 1:length(U.U)], [zeros(X.nX) for p = 1:length(U.U)],
                    similar(X.F[1]), IntegratedValueFunction(X), [zeros(X.nX) for p = 1:length(U.U)], zeros(X.nX,length(U.Θ)), verbose, K)

function Base.display{M<:NFXP}(est_res::EstimationResults{M})
    @printf "Results of estimation\n"
    @printf " * Method: NFXP\n"
    @printf " * loglikelihood: %s\n" loglikelihood(est_res)
    if length(join(Optim.minimizer(est_res.trace), ",")) < 40
       @printf " * Estimate: [%s]\n" join(Optim.minimizer(est_res.trace), ",")
       @printf " * Std.err.: [%s]\n" join(stderr(est_res), ",")
   else
       @printf " * Estimate: [%s, ...]\n" join(Optim.minimizer(est_res.trace)[1:2], ",")
       @printf " * Std.err.: [%s, ...]\n" join(stderr(est_res)[1:2], ",")
   end
   @printf " Iterations\n"
   @printf " * Maximum likelihood: %d\n" Optim.iterations(est_res.trace)
   @printf " * Contractions: %d\n" est_res.conv.iter[1]
   @printf " * Newton steps: %d\n" est_res.conv.iter[2]
end

type NFXPCacheVars
    Θᵏ
    last_Θᵏ # Not Θᵏ⁺¹, but the last parameter vector that was used to solve the model
    score
end

function NFXPCacheVars(nvar, n)
         Θᵏ = zeros(nvar)
    last_Θᵏ = zeros(nvar)
    score   = zeros(n, nvar)
    NFXPCacheVars(Θᵏ, last_Θᵏ, score)
end
