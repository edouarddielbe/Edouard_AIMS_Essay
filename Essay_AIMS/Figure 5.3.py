import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import networkx as nx

np.random.seed(42)

# Parameters
beta    = 0.3
gamma   = 0.1
D_I     = 0.01
M       = 3
N       = 1000
T_max   = 100.0
I0_frac = 0.1

A_meta   = D_I * np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=float)
D_meta   = np.diag(A_meta.sum(axis=1))
L_meta   = D_meta - A_meta
k_avg_values = [2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10]

##Build the ER networks
def build_ER_network(N, k_avg, seed=0):
    G   = nx.erdos_renyi_graph(N, k_avg/(N-1), seed=seed)
    A   = nx.to_numpy_array(G)
    adj = [list(G.neighbors(i)) for i in range(N)]
    return adj, A
# Gillespie
def gillespie_SIR(beta, gamma, A_meta, M, N, adjs, T_max, seed=3):
    state = [np.zeros(N, dtype=int) for _ in range(M)] 
    state[0][:max(1, int(0.10 * N))] = 1
    times = [0.0]
    I_counts = [[np.sum(s == 1) for s in state]]
    t = 0.0
    for _ in range(800_000):
        rates = []; events = []
        for mu in range(M):
            adj = adjs[mu]; s = state[mu]
            for i in range(N):
                if s[i] == 0:
                    inf_nb = sum(1 for j in adj[i] if s[j] == 1)
                    if inf_nb > 0:
                        rates.append(beta * inf_nb)
                        events.append(('infect', mu, i, None))
                elif s[i] == 1:
                    rates.append(gamma)
                    events.append(('recover', mu, i, None))
                    for nu in range(M):
                        if A_meta[mu, nu] > 0:
                            rates.append(A_meta[mu, nu])
                            events.append(('migrate', mu, i, nu))
        total_rate = sum(rates)
        if total_rate == 0:
            break
        t += np.random.exponential(1.0 / total_rate)
        if t > T_max: 
            break
        r = np.random.random() * total_rate
        cumsum = 0
        for idx, rate in enumerate(rates):
            cumsum += rate
            if r <= cumsum:
                etype, mu, i, nu = events[idx]
                break
        if etype == 'infect': state[mu][i] = 1
        elif etype == 'recover': state[mu][i] = 2
        elif etype == 'migrate':
            dest_S = np.where(state[nu] == 0)[0]
            if len(dest_S) > 0:
                state[mu][i] = 0
                state[nu][dest_S[np.random.randint(len(dest_S))]] = 1
        times.append(t)
        I_counts.append([np.sum(s == 1) for s in state])
    I_counts = np.array(I_counts, dtype=float) / N
    return np.array(times), I_counts.mean(axis=1)

#IBM ODE
def ode_IBM_SIR(t, X, beta, gamma, N, M, As, L_meta):
    X = np.clip(X, 0.0, 1.0)
    dX = np.zeros(3 * N * M)
    for mu in range(M):
        S_mu = X[mu*3*N : mu*3*N + N]
        I_mu = X[mu*3*N + N : mu*3*N + 2*N]
        infection = beta * S_mu * (As[mu] @ I_mu)
        recovery = gamma * I_mu
        mig_S = sum(L_meta[mu,nu]*X[nu*3*N : nu*3*N + N] for nu in range(M))
        mig_I = sum(L_meta[mu,nu]*X[nu*3*N + N : nu*3*N + 2*N] for nu in range(M))
        dX[mu*3*N : mu*3*N + N] = - infection - mig_S
        dX[mu*3*N + N : mu*3*N + 2*N] = +infection - recovery - mig_I
        dX[mu*3*N + 2*N: mu*3*N + 3*N] = +recovery
    return dX

def run_IBM(beta, gamma, N, M, As, L_meta, T_max):
    X0 = np.zeros(3 * N * M)
    for mu in range(M): X0[mu*3*N : mu*3*N + N] = 1.0
    X0[0:N] = 1.0 - I0_frac; X0[N:2*N] = I0_frac
    sol = solve_ivp(ode_IBM_SIR, [0, T_max], X0, t_eval=np.linspace(0, T_max, 500),
                    args=(beta, gamma, N, M, As, L_meta),method='RK45',  max_step=0.5)
    return sol.t, np.array([sol.y[mu*3*N+N : mu*3*N+2*N].mean(axis=0) for mu in range(M)]).mean(axis=0)

#Loop
k_vals = []; rmse_vals = []
for k_avg in k_avg_values:
    adjs, As = [], []
    for mu in range(M):
        adj, A = build_ER_network(N, k_avg, seed=mu+42)
        adjs.append(adj); As.append(A)
    t_gil, I_gil = gillespie_SIR(beta, gamma, A_meta, M, N, adjs, T_max)
    t_ibm, I_ibm = run_IBM(beta, gamma, N, M, As, L_meta, T_max)
    t_common = np.linspace(0, T_max, 300)
    rmse = np.sqrt(np.mean((np.interp(t_common, t_gil, I_gil) - np.interp(t_common, t_ibm, I_ibm))**2))
    k_vals.append(k_avg); rmse_vals.append(rmse) 

#Figure
plt.figure(figsize=(14, 7))
plt.plot(k_vals, np.log10(rmse_vals), 'bo-', lw=2.5, markersize=8,
markerfacecolor='white', markeredgewidth=2)
plt.xlabel(r'Mean degree $\langle k \rangle$', fontsize=12)
plt.ylabel(r'$\log_{10}[RSME(Gillespie - IBM)]$', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('Figure 5.3.png')
plt.show()
