fit(E::NFXP, U, S, D) = fit_nfxp(U, S, D, E)
fit_nfxp(U, S, D) = fit_nfxp(U, S, D, NFXP(U, S, D))
function fit_nfxp{Tu<:AbstractUtility, Ts<:Stocas.AbstractState}(U::Tu,
                  S::Ts,
                  D::Data,
                  E::NFXP;
                  ## Keyword arguments
                  # Convergence parameters
                  ε_outer = 1e-8,
                  ε_inner = 1e-11,
                  method = NewtonTrustRegion(),
                  options = nothing)
    options = options == nothing ? Optim.Options(show_trace=E.verbose, extended_trace=E.verbose, g_tol = ε_outer) : options

    conv_info = ConvergenceNFXP()

    # Cache variables
    c = NFXPCacheVars(U.nvar, D.nobs)
    c.Θᵏ .= U.Θ
    c.last_Θᵏ .= copy(c.Θᵏ)+1.

      llᵈ(x)           =   ll(x,                     c.last_Θᵏ, U, S, D, E, conv_info)
     ∇llᵈ(gradient, x) =  ∇ll!(gradient, c.score, x, c.last_Θᵏ, U, S, D, E, conv_info)
    ∇²llᵈ(hessian, x)  = ∇²ll!(hessian,  c.score, x, c.last_Θᵏ, U, S, D, E, conv_info)

    likelihood = TwiceDifferentiable(llᵈ, ∇llᵈ, ∇²llᵈ, c.Θᵏ)
    result = optimize(likelihood, c.Θᵏ, method, options)
    # Return gradient and hessian at estimated parameters
    gradient = zeros(U.nvar)
    hessian = zeros(U.nvar, U.nvar)
     ∇llᵈ(gradient, c.Θᵏ)
    ∇²llᵈ(hessian, c.Θᵏ)
    EstimationResults(E,
                      -Optim.minimum(result)*D.nobs,
                      -gradient*D.nobs,
                      -hessian*D.nobs,
                      likelihood,
                      Optim.minimizer(result),
                      conv_info,
                      result,
                      D.nobs,
                      likelihood)
end

""" Updates the choice specific utilities based on most recent parameters. """
update_utilities!(U) = error("Need to implement update_utilities! for $(typeof(U))")
function update_utilities!(U::LinearUtility)
    for i in 1:length(U.Z)
        U.U[i] .= U.Z[i]*U.Θ
    end
end
