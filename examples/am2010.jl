using MDPTools, NestedFixedPoint
#srand(122)

# 2 Model
bdisc = 0.95 # discount factor
truepar = [-1.9;1.;2.] # fixed costs, ..., entry costs
# State 1: Exogenous state
nX1 = 2
X1 = [0.;1.]
F1 = [sparse([1. 0.; 1. 0.]), sparse([0. 1.; 0. 1.])]

nX2 = 5
X2 = 1:nX2
M = [0.8 0.2 0.0 0.0 0.0;
	 0.2 0.6 0.2 0.0 0.0;
	 0.0 0.2 0.6 0.2 0.0;
	 0.0 0.0 0.2 0.6 0.2;
	 0.0 0.0 0.0 0.2 0.8;]

S = States(State(X1, F1),
           CommonState(X2, M))

Z = [zeros(nX2*2, 3), # don't enter
             [-ones(nX2) log(X2) -ones(nX2); # enter
			  -ones(nX2) log(X2)  zeros(nX2)]]

U = LinearUtility(Z, bdisc, truepar)

M, N, T = 2, 100, 1000
D = simulate(U, S, M, N, T)

n_transitions = zeros(nX2, nX2)
for n = 1:T*N-1
	if D.id[n] == D.id[n+1,1]
		n_transitions[D.xs[n,2], D.xs[n+1,2]] += 1
	end
end

estimated_F2 = n_transitions./sum(n_transitions,2)

estimated_S =  States(State(X1, F1), CommonState(X2, estimated_F2))

U.Θ*=0.0
am_nfxp = fit_nfxp(U, estimated_S, D)
