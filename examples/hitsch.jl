using MDPTools, NestedFixedPoint, Plots
srand(122)

n = [0, 2, 5]
next_i(i, k) = max(0, min(I, i+n[k]-1))

p = [0.6, 0.8, 1.] # FIX this
F_P = [0.15 0.0 0.85; 0.0 0.25 0.75; 0.10 0.10 0.80]
SP = CommonState(p, F_P)

I = 15
K = length(n)
F_i = [spzeros(I+1, I+1) for k = 1:K]
for i = 0:I, j = 1:K
        F_i[j][i+1, next_i(i, j)+1]=1
end

Si = State(0:I, F_i);

S = States(SP, Si)

delta, alpha = 3., 3.

Z1 = [[zeros(1); ones(I)] zeros(I+1) -next_i(0:I, 1);
      [zeros(1); ones(I)] zeros(I+1) -next_i(0:I, 1);
      [zeros(1); ones(I)] zeros(I+1) -next_i(0:I, 1)]
Z2 = [ones(3*(I+1)) -[p[1]*n[2]*ones(I+1);p[2]*n[2]*ones(I+1);p[3]*n[2]*ones(I+1)] -kron(ones(3), next_i(0:I, 2))]
Z3 = [ones(3*(I+1)) -[p[1]*n[3]*ones(I+1);p[2]*n[3]*ones(I+1);p[3]*n[3]*ones(I+1)] -kron(ones(3), next_i(0:I, 3))]
U = LinearUtility([Z1, Z2, Z3], 0.95, [delta; alpha; 0.05])

M, N, T = 1, 400, 3*12
D = simulate(U, S, M, N, T);

hitsch_nfxp = fit_nfxp(U, S, D)

theme(:dark)
plot(plot([U.P[i][1:16] for i = 1:3], labels = ["buy 0" "buy 2" "buy 5"], title = "Low price", xlabel = "inventory", ylabel = "probability", ylabel = "probability"),
            plot([U.P[i][17:32] for i = 1:3], labels = ["buy 0" "buy 2" "buy 5"], title = "Medium price", xlabel = "inventory", ylabel = "probability"),
            plot([U.P[i][33:end] for i = 1:3], labels = ["buy 0" "buy 2" "buy 5"], title = "High price", xlabel = "inventory", ylabel = "probability"), layout = (1,3), size = (900,300))
