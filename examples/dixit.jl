using MDPTools, NestedFixedPoint
srand(123)
# State 1
X1 = 0:1
F1 = [[1. 0.; 1. 0.], [0. 1.; 0. 1.]]

# State 2
X2 = 1:5
nX2 = length(X2)
F2 = 1./(1+abs(ones(length(X2),1)*X2'-X2*ones(1, length(X2))))
F2 = F2./sum(F2,1)'

# States
S = States(State(X1, F1), CommonState(X2, F2))

# Utility variables
Z1 = zeros(nX2*2, 3)
Z2 = [ones(nX2) X2 -ones(nX2);
      ones(nX2) X2 zeros(nX2)]

U = LinearUtility([Z1, Z2], 0.99, [-.1;.2;1])

sol = solve(U, S)
# 3 Simulate Data
T = 12*3
N = 20
M = 3
D = simulate(U, S, M, N, T)

# Estimate market probability matrix
F2hat = zeros(5, 5)
for n = 1:T*N
   if n%T != 0
      F2hat[ind2sub((5,2), D.x[n])[1],
            ind2sub((5,2), D.x[n + 1])[1]] += 1
   end
end

F2hat = F2hat./vec(sum(F2hat,2))

# States based on estimated transitions
estimated_S = States(State(X1, F1), CommonState(X2, F2hat))

# Fit using NFXP
dixit_nfxp = fit_nfxp(U, estimated_S, D)
