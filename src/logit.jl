function solve_if_new!(Θ, last_Θ, U, X, V, conv_info, alg=Stocas.Poly())
    if Θ ≢ last_Θ
        copy!(last_Θ, Θ)
        copy!(U.Θ, Θ)
        update_utilities!(U)
        conv_info.iter += solve!(U, X, V, Stocas.Poly())
        true
    end
    false
end

function ll(Θ, last_Θ, U, X, D, E, conv_info)
    solve_if_new!(Θ, last_Θ, U, X, E.V, conv_info, Stocas.Poly())
    ll(U, D, E)
end
function ll(U, D, E)
    t_ll = 0.
    for ia = 1:length(E.P)
        map!(log, E.logP[ia], U.P[ia])
    end
    @inbounds for ia = 1:length(U.P), (x, a) in zip(D.x, D.a)
        t_ll -= E.logP[ia][x]*(a == ia)
    end
    t_ll/length(D.a)
end

function ∇ll!(grad, score, Θ, last_Θ, U, X, D, E, conv_info)
    solve_if_new!(Θ, last_Θ, U, X, E.V, conv_info, Stocas.Poly())
    ∇ll!(grad, score, U, X, D, E)
end
function ∇ll!(grad, score, U, X, D, E::NFXP)
    sc_logit!(score, U, X, D, E)
    grad[:] = mean(score,1)
end

function sc_logit!(score, U, X, D, E::NFXP)
    E.dVu .= (I-E.V.βFP)\sum(U.P[a].*U.Z[a] for a = 1:length(E.P))
    score_model = [U.Z[a] + U.β*(X.F[a]*E.dVu) - E.dVu for a = 1:length(E.P)]
    @inbounds for j = 1:size(score, 2)
        for i = 1:D.nobs
            score[i, j] = -score_model[D.a[i]][D.x[i],j]
        end
    end
end

function ∇²ll!(hessian, scores, Θ, last_Θ, U, X, D, E, conv_info)
    if solve_if_new!(Θ, last_Θ, U, X, E.V, conv_info, Stocas.Poly())
        sc_logit!(scores, U, X, D, E)
    end
    hessian .= scores'*scores/size(scores, 1)
end
