import numpy as np
import scipy as sp
from scipy.optimize import minimize
from scipy.linalg import svd


class TooManyEvals(Exception):
    def __init__(self, message, best_x, best_fun):
        super().__init__(message)
        self.best_x = best_x
        self.best_fun = best_fun


def make_tracking_limited_fun(fun, max_calls):
    """
    Wraps a function to limit calls AND track the best input seen.
    Returns: (wrapped_function, getter_for_best)
    """
    state = {
        'n': 0, 
        'best_x': None, 
        'best_fun': float('inf')
    }

    def wrapped(x, *args, **kwargs):
        state['n'] += 1
        
        # 1. Call function
        val = fun(x, *args, **kwargs)
        
        # 2. Track best
        if val < state['best_fun']:
            state['best_fun'] = val
            state['best_x'] = x.copy()
            
        # 3. Check limit
        if state['n'] >= max_calls:
            raise TooManyEvals(
                f"Exceeded {max_calls} evaluations", 
                state['best_x'], 
                state['best_fun']
            )
        return val
    
    wrapped.state = state

    # Helper to retrieve data if no exception raised
    def get_best():
        return state['best_x'], state['best_fun']

    return wrapped, get_best



class HierarchicalOptimizer:
    def __init__(self, optimizer_object, params_0, pipeline=None, h_step=1e-5, reg=1e-6, print_flag=True):
        self.opt_obj = optimizer_object
        self.phi = np.array(params_0)
        self.h_step = h_step
        self.reg = reg
        
        # Subspace Basis Vectors
        self.Vs = None # Stiff
        self.Vl = None # Sloppy
        self.eigenvals = None
        
        # History for convergence checks
        self.Vs_prev = None
        self.phi_prev = None
        self.loss_prev = float('inf')
        
        # Tracking
        self.history = {'best_loss': [], 'iters': [], 'best_params': [], 'loss': [], 'delta_phi': [], 'delta_s': []}
        self.iter_calls = 0
        self.best_loss = float('inf')
        self.best_params = np.zeros_like(self.phi)
        
        self.print_flag = print_flag
        
        # --- Define Default Pipeline if None provided ---
        if pipeline is None:
            self.pipeline = [
                ('decompose_stochastic', {'k_samples': 10, 'percent_info': 0.95}),
                ('optimize_stiff',       {'max_iter': 40}),
                ('realign_sloppy',       {}),
                ('optimize_sloppy',      {'max_evals': 100}),
                ('check_convergence',    {'tol_s': 1e-3})
            ]
        else:
            self.pipeline = pipeline
    
    
    def _track_iters(self, loss, params):
        
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_params = params
        
        self.iter_calls += 1
        
        self.history['best_loss'].append(self.best_loss)
        self.history['iters'].append(self.iter_calls)
        self.history['best_params'].append(self.best_params)
        
        if self.print_flag:
            print("loss: ", loss, "iter: ", self.iter_calls)
        
    
    
    def _loss_wrapper(self, params):
        loss, _, _, _, _, _ = self.opt_obj.objective_function_diff_full(params)
        
        self._track_iters(loss, params)
        return loss


    # =========================================================================
    #  STEP 1: DECOMPOSITION METHODS
    # =========================================================================

    def decompose_exact(self, percent_info=0.95, tau=1e-4):
        """Standard N+1 evaluation decomposition."""
        print(f"\n[Step] Exact Decomposition (N={len(self.phi)})")
        # ... (Implementation of gradients N+1 evals) ...
        # (Simplified logic for brevity - insert your exact logic here)
        # For prototype, reusing the logic from previous messages:
        J_rows = []
        loss, r_base, _, _, _, _ = self.opt_obj.objective_function_diff_full(self.phi)
        r_base = np.array(r_base)
        
        self._track_iters(loss, self.phi)
        
        for i in range(len(self.phi)):
            p = self.phi.copy(); p[i] += self.h_step
            loss, r_new, _, _, _, _ = self.opt_obj.objective_function_diff_full(p)
            J_rows.append((np.array(r_new) - r_base) / self.h_step)
            
            self._track_iters(loss, p)
        
        self._process_hessian(np.array(J_rows).T, percent_info, tau)

    def decompose_stochastic(self, k_samples=10, percent_info=0.95, tau=1e-4):
        """Stochastic k+1 evaluation decomposition."""
        print(f"\n[Step] Stochastic Decomposition (k={k_samples})")
        # 1. Random Matrix
        Omega = np.random.normal(0, 1, size=(len(self.phi), k_samples))
        Omega, _ = np.linalg.qr(Omega)
        
        # 2. Sketch
        loss, r_base, _, _, _, _ = self.opt_obj.objective_function_diff_full(self.phi)
        
        self._track_iters(loss, self.phi)
        
        r_base = np.array(r_base)
        Y_cols = []
        for i in range(k_samples):
            p = self.phi + self.h_step * Omega[:, i]
            loss, r, _, _, _, _ = self.opt_obj.objective_function_diff_full(p)
            Y_cols.append((np.array(r) - r_base) / self.h_step)
            
            self._track_iters(loss, p)
        
        Y = np.array(Y_cols).T
        F_small = Y.T @ Y
        self._process_hessian_stochastic(F_small, Omega, percent_info, tau)


    def _process_hessian_stochastic(self, F_small, Omega, percent_info, tau):
        # Internal helper to solve eigenproblem and lift vectors
        F_reg = F_small + self.reg * np.eye(F_small.shape[0])
        evals, evecs = np.linalg.eigh(F_reg)
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs_lifted = Omega @ evecs[:, idx] # Lift back to N dims
        self._threshold_and_split(evals, evecs_lifted, percent_info, tau)


    def _process_hessian(self, J, percent_info, tau):
        # Internal helper for exact Hessian
        F = J.T @ J
        evals, evecs = np.linalg.eigh(F + self.reg * np.eye(F.shape[0]))
        idx = np.argsort(evals)[::-1]
        self._threshold_and_split(evals[idx], evecs[:, idx], percent_info, tau)


    def _threshold_and_split(self, evals, evecs, percent_info, tau):
        total = evals.sum()
        ks = int(np.sum(np.cumsum(evals) < percent_info * total)) + 1
        ks = min(ks, len(evals))
        
        if ks < len(evals):
            # Ratio of first sloppy to last stiff? Or first sloppy to first sloppy?
            # Your logic: compare sloppy eigenvalues relative to the cutoff
            kl = int(np.count_nonzero((evals[ks:] / (evals[ks]+1e-16)) >= tau))
        else:
            kl = 0
            
        self.Vs = evecs[:, :ks]
        self.Vl = evecs[:, ks:ks+kl]
        print(f"  -> Split: Stiff={ks}, Sloppy={kl}")



    # =========================================================================
    #  STEP 2 & 4: OPTIMIZATION METHODS
    # =========================================================================

    def optimize_stiff(self, max_iter=40, bound_range=None):
        """
        Optimizes strictly within the identified stiff subspace.
        
        Args:
            max_iter (int): Maximum iterations for the Powell optimizer.
            bound_range (float or None): If provided, constrains the coefficients 'w' 
                                         to be within [-bound_range, bound_range].
                                         If None, optimization is unconstrained.
        """
        if self.Vs is None or self.Vs.shape[1] == 0: 
            return
            
        print(f"\n[Step] Stiff Opt (Powell) Dim:{self.Vs.shape[1]}")
        
        # 1. Define bounds if requested
        # Powell requires a list of (min, max) tuples for bounds
        if bound_range is not None:
            bounds = [(-bound_range, bound_range) for _ in range(self.Vs.shape[1])]
        else:
            bounds = None  # Scipy interprets None as "no bounds"

        # 2. Objective function for the subspace
        def reduced(w):
            # w contains the coefficients for the basis vectors in Vs
            # theta_new = theta_current + Vs * w
            loss = self._loss_wrapper(np.abs(self.phi + self.Vs @ w))
            return loss
        
        # 3. Run Optimization
        # We start at w=0 (which represents the current parameter set self.phi)
        res = minimize(reduced, 
                       np.zeros(self.Vs.shape[1]), 
                       method='Powell', 
                       bounds=bounds,  # Pass the bounds (or None) here
                       options={'maxiter': max_iter, 
                                'maxfev': max_iter, 
                                'xtol': 1e-4, 
                                'ftol': 1e-4})
        
        # 4. Update Parameters
        self.phi = np.abs(self.phi + self.Vs @ res.x)
        self.history['loss'].append(res.fun)
        
        status = "Bounded" if bounds else "Unbounded"
        print(f"  -> Loss: {res.fun:.6f} ({status})")



    def optimize_sloppy(self, max_iter=150):
        if self.Vl is None or self.Vl.shape[1] == 0: return
        print(f"\n[Step] Sloppy Opt (Nelder-Mead) Dim:{self.Vl.shape[1]}")
        
        def reduced(w):
            loss = self._loss_wrapper(np.abs(self.phi + self.Vl @ w))
            return loss
        
        wrapped, get_best = make_tracking_limited_fun(reduced, max_iter)
        
        try:
            minimize(wrapped, np.zeros(self.Vl.shape[1]), method="Nelder-Mead", options={'fatol': 1e-5, 'xatol': 1e-5}, tol=5e-5)
            w_best, f_best = get_best()
        except TooManyEvals as e:
            print("  -> Max evals reached.")
            w_best, f_best = e.best_x, e.best_fun
            if w_best is None: w_best = np.zeros(self.Vl.shape[1])

        self.phi = np.abs(self.phi + self.Vl @ w_best)
        self.history['loss'].append(f_best)
        print(f"  -> Loss: {f_best:.6f}")



    def optimize_sloppy_adaptive(self, max_iter=150, stall_tol=1e-5):
        """
        The Core Logic: Run Nelder-Mead -> Stall -> Realign -> Restart
        """
        if self.Vl is None or self.Vl.shape[1] == 0: return
        print(f"\n[Step 3] Adaptive Sloppy Optimization (Budget: {max_iter})...")
        
        evals_remaining = max_iter
        
        while evals_remaining > 20 and self.Vl.shape[1] > 0:
            
            # --- A. Run Nelder-Mead on CURRENT Subspace ---
            current_limit = min(60, evals_remaining)
            print(f"  -> Sub-run (Dim {self.Vl.shape[1]}) for {current_limit} evals...")
            
            def reduced(w):
                loss = self._loss_wrapper(np.abs(self.phi + self.Vl @ w))
                return loss
            
            wrapped, get_best = make_tracking_limited_fun(reduced, current_limit)
            
            try:
                res = minimize(wrapped, np.zeros(self.Vl.shape[1]), method="Nelder-Mead", options={'fatol': stall_tol})
                w_best = res.x; f_best = res.fun
                status = "Converged"
            except TooManyEvals as e:
                w_best = e.best_x; f_best = e.best_fun
                status = "Limit Hit"
            
            # Update State
            self.phi = np.abs(self.phi + self.Vl @ w_best)
            self.history['loss'].append(f_best)
            
            used = wrapped.state['n']
            
            evals_remaining -= used
            print(f"     Result: {status}, Loss: {f_best:.6f}")
            
            # If we converged fully or ran out of budget
            if evals_remaining < 10: break
            
            # --- B. Realign & Deflate (The Rotation) ---
            print("  -> Realigning Subspace to find new Stiff directions...")
            
            # Compute Reduced Hessian (Cost: k evals)
            loss, r_base, _, _, _, _ = self.opt_obj.objective_function_diff_full(self.phi)
            self._track_iters(loss, self.phi)
            
            
            r_base = np.array(r_base)
            J_cols = []
            
            for i in range(self.Vl.shape[1]):
                p = self.phi + self.h_step * self.Vl[:, i]
                loss, r, _, _, _, _ = self.opt_obj.objective_function_diff_full(p)
                
                self._track_iters(loss, p)
                
                J_cols.append((np.array(r) - r_base) / self.h_step)
                evals_remaining -= 1
            
            J_red = np.array(J_cols).T
            F_small = J_red.T @ J_red
            
            # Diagonalize
            evals, evecs = np.linalg.eigh(F_small)
            idx = np.argsort(evals)[::-1]
            evals = evals[idx]
            evecs = evecs[:, idx]
            
            # ROTATE
            self.Vl = self.Vl @ evecs[:,:]



    # =========================================================================
    #  STEP 3: REALIGNMENT
    # =========================================================================

    def realign_sloppy(self):
        if self.Vl is None or self.Vl.shape[1] == 0: return
        print("\n[Step] Realigning Sloppy Space")
        
        loss, r_base, _, _, _, _ = self.opt_obj.objective_function_diff_full(self.phi)
        r_base = np.array(r_base)
        J_cols = []
        
        self._track_iters(loss, self.phi)
        
        for i in range(self.Vl.shape[1]):
            p = self.phi + self.h_step * self.Vl[:, i]
            loss, r, _, _, _, _ = self.opt_obj.objective_function_diff_full(p)
            J_cols.append((np.array(r) - r_base) / self.h_step)
            
            self._track_iters(loss, p)
            
        J_red = np.array(J_cols).T
        evals, evecs = np.linalg.eigh(J_red.T @ J_red)
        
        # Sort Stiffest -> Sloppiest for Nelder Mead
        idx = np.argsort(evals)[::-1]
        self.Vl = self.Vl @ evecs[:, idx]
        print(f"  -> Realigned. Top Eigenvals: {evals[idx][:3]}")

    # =========================================================================
    #  STEP 5: CONVERGENCE CHECK
    # =========================================================================

    def check_convergence(self, tol_s=1e-3, tol_phi=1e-4, tol_loss=1e-5):
        if self.Vs_prev is None: return False
        
        # Metric A: Subspace Rotation
        min_dim = min(self.Vs.shape[1], self.Vs_prev.shape[1])
        if min_dim > 0:
            M = self.Vs[:, :min_dim].T @ self.Vs_prev[:, :min_dim]
            delta_s = max(1 - svd(M, compute_uv=False))
        else:
            delta_s = 0.0

        # Metric B & C
        delta_phi = np.linalg.norm(self.phi - self.phi_prev)
        delta_loss = abs(self.loss_prev - self.history['loss'][-1])
        
        self.history['delta_s'].append(delta_s)
        self.history['delta_phi'].append(delta_phi)
        
        print(f"  -> Conv Check: dS={delta_s:.1e}, dPhi={delta_phi:.1e}, dLoss={delta_loss:.1e}")
        
        if delta_s < tol_s and delta_phi < tol_phi: return True
        return False


    # =========================================================================
    #  RUN ENGINE
    # =========================================================================

    def run(self, max_cycles=50, max_iter=1300):
        print(f"Starting Pipeline with {len(self.pipeline)} steps...")
        
        for i in range(max_cycles):
            print(f"\n{'='*20} CYCLE {i+1} {'='*20}")
            
            # Snapshot start of cycle
            self.phi_prev = self.phi.copy()
            self.Vs_prev = self.Vs.copy() if self.Vs is not None else None
            self.loss_prev = self._loss_wrapper(self.phi) if not self.history['loss'] else self.history['loss'][-1]
            
            # Execute Pipeline
            should_break = False
            for step_name, kwargs in self.pipeline:
                # Dynamically call the method by name
                if hasattr(self, step_name):
                    method = getattr(self, step_name)
                    result = method(**kwargs)
                    
                    # If the method was check_convergence, it returns a boolean
                    if step_name == 'check_convergence' and result is True:
                        should_break = True
                        break
                else:
                    print(f"Warning: Method {step_name} not found!")
            
            if should_break:
                print("\n*** Convergence Criteria Met ***")
                break
            
            
            if self.iter_calls > max_iter:
                print("\n*** Number Maximum of Function Evaluations Reached ***")
                break