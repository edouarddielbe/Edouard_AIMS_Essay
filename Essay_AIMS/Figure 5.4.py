import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import networkx as nx

np.random.seed(3)

# Parameters
gamma   = 0.1
D_I     = 0.01
M       = 3
N       = 2000
T_max   = 100.0
I0_frac = 0.01
k_avg   = 6

beta_values = np.linspace(0.005, 0.08, 20)

A_meta   = D_I * np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=float)
D_meta   = np.diag(A_meta.sum(axis=1))
L_meta   = D_meta - A_meta
D_mu_vec = np.diag(L_meta)
pi_mu0   = 1.0 / M
D_mu0    = D_mu_vec[0]

# Build Erdos-Renyi Network
adjs, As, degrees_all = [], [], []
for mu in range(M):
    G   = nx.erdos_renyi_graph(N, k_avg/(N-1), seed=mu+3)
    A   = nx.to_numpy_array(G)
    adj = [list(G.neighbors(i)) for i in range(N)]
    deg = np.array([d for _, d in G.degree()])
    adjs.append(adj); As.append(A); degrees_all.append(deg)

rho_A      = float(np.max(np.abs(np.linalg.eigvalsh(As[0]))))
beta_c= (gamma + D_mu0) / (pi_mu0 * rho_A)

# Gillespie
def gillespie_final_size(beta, gamma, A_meta, M, N, adjs, T_max, seed=3):
    state = [np.zeros(N, dtype=int) for _ in range(M)]
    state[0][:max(1, int(I0_frac * N))] = 1
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
        if total_rate == 0: break
        t += np.random.exponential(1.0 / total_rate)
        if t > T_max: break
        r = np.random.random() * total_rate
        cumsum = 0
        for idx, rate in enumerate(rates):
            cumsum += rate
            if r <= cumsum:
                etype, mu, i, nu = events[idx]
                break
        if etype == 'infect':    state[mu][i] = 1
        elif etype == 'recover': state[mu][i] = 2
        elif etype == 'migrate':
            dest_S = np.where(state[nu] == 0)[0]
            if len(dest_S) > 0:
                state[mu][i] = 0
                state[nu][dest_S[np.random.randint(len(dest_S))]] = 1
    total_S = sum(np.sum(s == 0) for s in state)
    return 1.0 - total_S / (N * M)

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

def IBM_final_size(beta, gamma, N, M, As, L_meta, T_max):
    X0 = np.zeros(3 * N * M)
    for mu in range(M): X0[mu*3*N : mu*3*N + N] = 1.0
    X0[0:N] = 1.0 - I0_frac; X0[N:2*N] = I0_frac
    sol = solve_ivp(ode_IBM_SIR, [0, T_max], X0, t_eval=[T_max],
                    args=(beta, gamma, N, M, As, L_meta),
                    method='RK45', rtol=1e-5, atol=1e-7, max_step=0.5)
    total_S = sum(sol.y[mu*3*N : mu*3*N + N, -1].sum() for mu in range(M))
    return 1.0 - total_S / (N * M)

fs_gil = []
fs_ibm = []
for beta in beta_values:
    print(f"beta={beta:.4f} ...", end=' ', flush=True)
    fs_gil.append(gillespie_final_size(beta, gamma, A_meta, M, N, adjs, T_max, seed=3))
    fs_ibm.append(IBM_final_size(beta, gamma, N, M, As, L_meta, T_max))
    print(f"Gil={fs_gil[-1]:.3f}  IBM={fs_ibm[-1]:.3f}")

fs_gil = np.array(fs_gil)
fs_ibm = np.array(fs_ibm)

# WE Find the empirical beta_G from Gillespie
threshold  = 0.05
beta_G = None
for i in range(len(beta_values)-1):
    if fs_gil[i] < threshold and fs_gil[i+1] >= threshold:
        beta_G = beta_values[i]
        break

# Figure
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(beta_values, fs_gil, color='#16a34a', lw=2.5, marker='o',
        markersize=5, label='Gillespie')
ax.plot(beta_values, fs_ibm, color='#2563eb', lw=2.0, ls='--', marker='s',
        markersize=5, label='IBM')
ax.axvline(beta_c, color='#2563eb', lw=1.2, ls=':',
           label=f'$\\beta_c = {beta_c:.3f}$')
if beta_G is not None:
    ax.axvline(beta_G, color='#16a34a', lw=1.2, ls=':',
               label=f'$\\beta_G \\approx {beta_G:.3f}$')

ax.set_xlabel(r'Transmission rate $\beta$', fontsize=12)
ax.set_ylabel(r'$1-S$', fontsize=12)
ax.legend(fontsize=9, facecolor='white', edgecolor='#cccccc')
ax.grid(True, alpha=0.3)
ax.set_xlim(beta_values[0], beta_values[-1])
ax.set_ylim(-0.02, 1.05)
plt.savefig('Figure 5.4.png')
plt.show()
