using MDPTools, NestedFixedPoint
using DataFrames

# 1 Data
df = readtable(Pkg.dir("StructuralBase"*"/data/data1234.csv"))
n = 90
omax = 450000
Xval = linspace(0, omax/1000, n)
x = ceil(df[:odo]*n/omax)
data = DataFrame()
data[:a] = df[:replace]
data[:x] = x
data[:dx] = x[1:end]-[0;x[1:end-1]]
for i = eachindex(data[:dx])
    data[:dx][i] = data[:dx][i]*(1-data[:a][i])+data[:x][i]*data[:a][i]
end
remove_first_row_index=df[:id]-[0;df[:id][1:end-1]]
data=data[remove_first_row_index .== 0,:]
data[:a]=[data[:a][2:end];0]
maxi = Int64(maximum(data[:dx]))

# 2 Model
#function transition(p)

p=zeros(maxi+1);
for i = 1:1:maxi+1
    p[i]=sum(data[:dx] .== i-1)/length(data[:dx])
end

F = zeros(n, n)
for i = 1:n-length(p)+1
    F[i,i:i+length(p)-1] = p
end
for i=1:length(p)-1
    F[end-length(p)+i+1,end-length(p)+i+1:end]=[p[1:end-i-1];sum(p[end-i:end])]
end
F = [F, F[ones(Int64, n),:]]

S = State(Xval, F)

Z1 = [zeros(n) -0.001*Xval]
Z2 = [-ones(n) zeros(n)]

U = LinearUtility([Z1, Z2], 0.9999, [11., 3.])

a = vec(round(Int64,data[:a]))
x = vec(round(Int64,data[:x]))

D = Data(a, a+1, x)
U.Θ*=0
rust_nfxp = fit_nfxp(U, S, D; verbose = false);
