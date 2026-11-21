"""
spatial_cv/optimizers.py
Phase 4: Genetic Algorithm (GA) and Particle Swarm Optimization (PSO) for Ensemble Weight Optimization.
Vectorized, self-contained, reproducible implementations using NumPy.
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np

def optimize_weights_slsqp(preds_matrix, y_true):
    """
    Standard convex optimization with SLSQP:
    minimize MAE(y_true, sum(w_i * preds_i))
    subject to w_i >= 0 and sum(w_i) == 1.
    """
    from scipy.optimize import minimize
    n_models = preds_matrix.shape[1]
    
    def loss(w):
        w_norm = w / np.sum(w)
        pred = np.dot(preds_matrix, w_norm)
        return np.mean(np.abs(y_true - pred))
    
    w0 = np.full(n_models, 1.0 / n_models)
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    
    res = minimize(loss, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    w_opt = res.x / np.sum(res.x)
    return w_opt, res.fun

def optimize_weights_composite_slsqp(preds_matrix, y_true, alpha=0.7):
    """
    Multi-metric convex simplex optimization:
    minimize alpha * MAE + (1 - alpha) * RMSE
    subject to w_i >= 0 and sum(w_i) == 1.
    Balances peak outbreak calibration (RMSE) with epidemiological fidelity (MAE).
    """
    from scipy.optimize import minimize
    n_models = preds_matrix.shape[1]
    
    def loss(w):
        w_norm = w / np.sum(w)
        pred = np.dot(preds_matrix, w_norm)
        diff = y_true - pred
        mae = np.mean(np.abs(diff))
        rmse = np.sqrt(np.mean(diff ** 2))
        return alpha * mae + (1.0 - alpha) * rmse
    
    w0 = np.full(n_models, 1.0 / n_models)
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    
    res = minimize(loss, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    w_opt = res.x / np.sum(res.x)
    return w_opt, res.fun

def optimize_weights_ga(preds_matrix, y_true, pop_size=30, n_gen=35, seed=42):
    """
    Phase 4: Genetic Algorithm (GA) for simplex-constrained weight optimization.
    """
    rng = np.random.default_rng(seed)
    n_models = preds_matrix.shape[1]

    # Initialize population on the simplex
    pop = rng.dirichlet(np.ones(n_models), size=pop_size)
    
    def fitness(w):
        pred = np.dot(preds_matrix, w)
        return -np.mean(np.abs(y_true - pred)) # maximize negative MAE

    scores = np.array([fitness(ind) for ind in pop])

    for gen in range(n_gen):
        # Elitism: retain top 2
        elite_idx = np.argsort(scores)[-2:]
        new_pop = [pop[i].copy() for i in elite_idx]

        # Tournament selection & crossover
        while len(new_pop) < pop_size:
            p1_idx = rng.choice(pop_size, 3)
            p1 = pop[p1_idx[np.argmax(scores[p1_idx])]]
            p2_idx = rng.choice(pop_size, 3)
            p2 = pop[p2_idx[np.argmax(scores[p2_idx])]]

            # Arithmetic crossover
            alpha = rng.uniform(0.2, 0.8)
            child = alpha * p1 + (1.0 - alpha) * p2

            # Mutation: Gaussian perturbation projected to simplex
            if rng.random() < 0.3:
                child += rng.normal(0, 0.08, size=n_models)
                child = np.clip(child, 0.001, None)
            child /= np.sum(child)
            new_pop.append(child)

        pop = np.array(new_pop)
        scores = np.array([fitness(ind) for ind in pop])

    best_idx = np.argmax(scores)
    best_weights = pop[best_idx]
    best_mae = -scores[best_idx]
    return best_weights, best_mae

def optimize_weights_pso(preds_matrix, y_true, n_particles=30, n_iters=35, seed=42):
    """
    Phase 4: Particle Swarm Optimization (PSO) for simplex-constrained weight optimization.
    """
    rng = np.random.default_rng(seed)
    n_models = preds_matrix.shape[1]

    # Initialize positions on the simplex and velocities
    X_pos = rng.dirichlet(np.ones(n_models), size=n_particles)
    V_vel = rng.normal(0, 0.05, size=(n_particles, n_models))

    def evaluate(w):
        pred = np.dot(preds_matrix, w)
        return np.mean(np.abs(y_true - pred))

    pbest_pos = X_pos.copy()
    pbest_val = np.array([evaluate(ind) for ind in X_pos])

    gbest_idx = np.argmin(pbest_val)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]

    w_inertia = 0.7
    c1, c2 = 1.4, 1.4

    for it in range(n_iters):
        r1 = rng.uniform(0, 1, size=(n_particles, n_models))
        r2 = rng.uniform(0, 1, size=(n_particles, n_models))

        V_vel = w_inertia * V_vel + c1 * r1 * (pbest_pos - X_pos) + c2 * r2 * (gbest_pos - X_pos)
        X_pos += V_vel

        # Project back to simplex
        X_pos = np.clip(X_pos, 0.0001, None)
        X_pos /= np.sum(X_pos, axis=1, keepdims=True)

        # Update pbest and gbest
        curr_vals = np.array([evaluate(ind) for ind in X_pos])
        improved = curr_vals < pbest_val
        pbest_pos[improved] = X_pos[improved]
        pbest_val[improved] = curr_vals[improved]

        if np.min(curr_vals) < gbest_val:
            gbest_idx = np.argmin(curr_vals)
            gbest_pos = X_pos[gbest_idx].copy()
            gbest_val = curr_vals[gbest_idx]

    return gbest_pos, gbest_val
