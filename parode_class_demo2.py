# parode_class_demo.py
# ParODE: GPU-parallel fixed-point ODE solver 
# ParODE2: GPU-parallel fixed-point ODE solver (PDE-residual augmented)
# Author: Martin Veselsky 
# Date: 2025

import torch, time

class ParODE:
    def __init__(self, M=10, N=10, K=200000, ic_weight=1e3, reg_lambda=1e-8,
                 max_newton_iters=200, newton_tol=1e-10, damping=0.8,
                 device=None, dtype=torch.float64):
        """
        Initialize ParODE solver.
        Args:
            M, N          : polynomial basis orders in y and t
            K             : number of random collocation samples
            ic_weight     : weight for enforcing F(y0,t0)=y0
            reg_lambda    : Tikhonov regularization parameter
            max_newton_iters, newton_tol, damping : Newton solver controls
            device, dtype : torch device and dtype
        """
        self.M, self.N, self.K = M, N, K
        self.ic_weight = ic_weight
        self.reg_lambda = reg_lambda
        self.max_newton_iters = max_newton_iters
        self.newton_tol = newton_tol
        self.damping = damping
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

    def _build_basis(self, ys, ts):
        """Build polynomial basis and its partial derivatives."""
        M, N, K = self.M, self.N, self.K
        Phi = torch.stack([ys**m for m in range(M)], dim=1)
        Tpow = torch.stack([ts**n for n in range(N)], dim=1)
        dPsi = torch.zeros((K, N), device=self.device, dtype=self.dtype)
        if N > 1:
            for n in range(1, N):
                dPsi[:, n] = n * ts**(n-1)
        dPhi = torch.zeros((K, M), device=self.device, dtype=self.dtype)
        if M > 1:
            for m in range(1, M):
                dPhi[:, m] = m * ys**(m-1)
        return Phi, Tpow, dPhi, dPsi

    def _assemble_matrix(self, ys, ts, s_k, Phi, Tpow, dPhi, dPsi):
        """Assemble collocation matrix B."""
        M, N, K = self.M, self.N, self.K
        B = torch.zeros((K, M*N), device=self.device, dtype=self.dtype)
        col = 0
        for m in range(M):
            pm = Phi[:, m]
            dpm = dPhi[:, m]
            for n in range(N):
                B[:, col] = pm * dPsi[:, n] + dpm * Tpow[:, n] * s_k
                col += 1
        return B

    def _evaluate_F_and_dFdy(self, y_vec, t_vec, A):
        """Evaluate F(y,t) and dF/dy for vectors y_vec,t_vec."""
        M, N = self.M, self.N
        y_pows = torch.stack([y_vec**m for m in range(M)], dim=1)
        t_pows = torch.stack([t_vec**n for n in range(N)], dim=1)
        YT = y_pows.unsqueeze(2) * t_pows.unsqueeze(1)
        F = (YT * A.unsqueeze(0)).sum(dim=(1,2))
        if M > 1:
            y_pows_m1 = torch.stack([y_vec**(m-1) if m>=1 else torch.zeros_like(y_vec)
                                     for m in range(M)], dim=1)
            m_idx = torch.arange(M, device=A.device, dtype=A.dtype).unsqueeze(1)
            mA = (m_idx * A)
            dF = (y_pows_m1.unsqueeze(2) * t_pows.unsqueeze(1) * mA.unsqueeze(0)).sum(dim=(1,2))
        else:
            dF = torch.zeros_like(y_vec)
        return F, dF

    def solve(self, func, y0, t_grid):
        """
        Solve ODE dy/dt = func(y,t) from t_grid[0] to t_grid[-1].
        Returns y(t_grid) predicted by the fixed-point polynomial approach.
        """
        device, dtype = self.device, self.dtype
        y0 = torch.as_tensor(y0, device=device, dtype=dtype)
        t0, tf = t_grid[0].item(), t_grid[-1].item()
        print(f"Device: {device}, dtype: {dtype}")
        print(f"Basis: {self.M} x {self.N}, samples K={self.K}")

        # --- Collocation sampling ---
        torch.manual_seed(0)
        delta_y = 0.8
        ys_samples = (y0.item() - delta_y) + 2*delta_y * torch.rand(self.K, device=device, dtype=dtype)
        ts_samples = t0 + (tf - t0) * torch.rand(self.K, device=device, dtype=dtype)
        s_k = func(ys_samples, ts_samples)

        # --- Basis and collocation matrix ---
        Phi, Tpow, dPhi, dPsi = self._build_basis(ys_samples, ts_samples)
        B = self._assemble_matrix(ys_samples, ts_samples, s_k, Phi, Tpow, dPhi, dPsi)
        BTB = B.T @ B
        rhs = B.T @ s_k

        # --- Add initial condition constraint F(y0,t0)=y0 ---
        phi_ic = torch.zeros((self.M*self.N,), device=device, dtype=dtype)
        col = 0
        for m in range(self.M):
            for n in range(self.N):
                phi_ic[col] = (y0**m) * (t0**n)
                col += 1
        BTB = BTB + self.ic_weight * (phi_ic.unsqueeze(1) @ phi_ic.unsqueeze(0))
        rhs = rhs + self.ic_weight * phi_ic * y0

        # --- Regularize and solve for coefficients a ---
        I = torch.eye(self.M*self.N, device=device, dtype=dtype)
        a = torch.linalg.solve(BTB + self.reg_lambda * I, rhs)
        A = a.view(self.M, self.N)

        # --- Newton fixed-point iteration (fully parallel over t_grid) ---
        L = len(t_grid)
        y_batch = torch.full((L,), y0.item(), device=device, dtype=dtype)
        start = time.time()
        for it in range(self.max_newton_iters):
            Fv, dFdy = self._evaluate_F_and_dFdy(y_batch, t_grid, A)
            G = Fv - y_batch
            dG = dFdy - 1.0
            denom = torch.where(dG.abs() < 1e-12, torch.sign(dG)*1e-12 + 1e-12, dG)
            delta = G / denom
            y_new = y_batch - self.damping * delta
            max_change = torch.max(torch.abs(y_new - y_batch)).item()
            rms_resid = torch.sqrt(torch.mean(G**2)).item()
            y_batch = y_new
            if (it % 10) == 0 or it == 0:
                print(f"Iter {it:3d}: max_change={max_change:.3e}, rms_resid={rms_resid:.3e}")
            if max_change < self.newton_tol:
                print(f"Converged at iter {it}, max_change={max_change:.3e}")
                break
        print(f"Newton loop time: {time.time()-start:.3f}s")

        return y_batch, A

    def compare_to_rk4(self, func, y0, t_grid, y_pred):
        """Compare the ParODE solution to RK4 reference and print diagnostics."""    
        def rk4_integrate(f, y0, t_grid):
            y = y0.clone()
            ys = [y.clone()]
            for i in range(len(t_grid) - 1):
                t = t_grid[i]
                dt = t_grid[i + 1] - t_grid[i]
                k1 = f(y, t)
                k2 = f(y + 0.5 * dt * k1, t + 0.5 * dt)
                k3 = f(y + 0.5 * dt * k2, t + 0.5 * dt)
                k4 = f(y + dt * k3, t + dt)
                y = y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
                ys.append(y.clone())
            return torch.stack(ys, dim=0)
    
        y_ref = rk4_integrate(func, torch.tensor(y0, device=self.device, dtype=self.dtype), t_grid)
        rel_err = torch.norm(y_ref - y_pred) / torch.norm(y_ref)
        dt = t_grid[1] - t_grid[0]
        dydt_fd = torch.empty_like(y_pred)
        dydt_fd[:-1] = (y_pred[1:] - y_pred[:-1]) / dt
        dydt_fd[-1] = dydt_fd[-2]
        residual = torch.norm(dydt_fd - func(y_pred, t_grid))

        print("\nDiagnostics vs RK4:")    
        print(f"Relative L2 error: {rel_err.item():.3e}")
        print(f"Residual norm ||d/dt F - f||: {residual.item():.3e}")

        print("\n t     RK4_ref     F_fixedpt")
        for i in range(0, min(10, len(t_grid))):
            print(f"{t_grid[i].item():6.3f} {y_ref[i].item():16.10e} {y_pred[i].item():16.10e}")
    
        return rel_err.item(), residual.item()

class ParODE2:
    def __init__(self, M=10, N=10, K=200000, ic_weight=1e3, reg_lambda=1e-8,
                 max_newton_iters=200, newton_tol=1e-10, damping=0.8,
                 device=None, dtype=torch.float64):
        self.M, self.N, self.K = int(M), int(N), int(K)
        self.ic_weight = float(ic_weight)
        self.reg_lambda = float(reg_lambda)
        self.max_newton_iters = int(max_newton_iters)
        self.newton_tol = float(newton_tol)
        self.damping = float(damping)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

    def _sample_collocation(self, y0, t0, tf):
        """Return ys_samples (K,), ts_samples (K,)."""
        # deterministic seed for reproducibility
        torch.manual_seed(0)
        delta_y = 0.8
        ys = (float(y0) - delta_y) + 2.0 * delta_y * torch.rand(self.K, device=self.device, dtype=self.dtype)
        ts = float(t0) + (float(tf) - float(t0)) * torch.rand(self.K, device=self.device, dtype=self.dtype)
        return ys, ts

    def _build_power_matrices(self, ys, ts):
        """
        Build y_pows: (K, M) with y^m and t_pows: (K, N) with t^n,
        and their partial derivatives dPhi (K,M) and dTpow (K,N) for m>=1,n>=1.
        """
        M, N = self.M, self.N
        # powers
        y_pows = torch.stack([ys**m for m in range(M)], dim=1)   # (K, M)
        t_pows = torch.stack([ts**n for n in range(N)], dim=1)   # (K, N)
        # derivatives: d/dy y^m = m*y^(m-1), with m=0 -> 0
        dPhi = torch.zeros_like(y_pows)
        if M > 1:
            m_idx = torch.arange(M, device=self.device, dtype=self.dtype)
            # for m>=1: m * y^(m-1)
            dPhi[:, 1:] = (m_idx[1:].unsqueeze(0) * y_pows[:, :-1])
        # derivatives: d/dt t^n = n*t^(n-1), n=0 -> 0
        dTpow = torch.zeros_like(t_pows)
        if N > 1:
            n_idx = torch.arange(N, device=self.device, dtype=self.dtype)
            dTpow[:, 1:] = (n_idx[1:].unsqueeze(0) * t_pows[:, :-1])
        return y_pows, t_pows, dPhi, dTpow

    def _assemble_pde_matrix(self, y_pows, t_pows, dPhi, dTpow, f_k):
        """
        Build PDE collocation matrix P (K, M*N) where each row is:
          row = (partial_t basis flattened) + f_k * (partial_y basis flattened)
        and RHS b_pde = f_k.
        Implementation is fully vectorized using broadcasting.
        """
        K = y_pows.shape[0]
        M, N = self.M, self.N

        # B_dt entries: y^m * (n * t^{n-1})  -> shape (K, M, N)
        B_dt = (y_pows.unsqueeze(2) * dTpow.unsqueeze(1))   # (K, M, N)

        # B_dy entries: (m * y^{m-1}) * t^n -> shape (K, M, N)
        B_dy = (dPhi.unsqueeze(2) * t_pows.unsqueeze(1))    # (K, M, N)

        # flatten columns in same ordering as A.view(M,N): m-major then n
        P = (B_dt + f_k.unsqueeze(1).unsqueeze(1) * B_dy).reshape(K, M * N)  # (K, M*N)

        b = f_k  # (K,)
        return P, b

    def _assemble_ic_row(self, y0, t0):
        """Return phi_ic (M*N,) with entries y0^m * t0^n"""
        M, N = self.M, self.N
        phi_ic = torch.empty((M * N,), device=self.device, dtype=self.dtype)
        col = 0
        for m in range(M):
            ym = (y0**m)
            for n in range(N):
                phi_ic[col] = ym * (t0**n)
                col += 1
        return phi_ic

    def _evaluate_F_and_dFdy(self, y_vec, t_vec, A):
        """Vectorized eval of F(y,t) and dF/dy for vectors y_vec,t_vec and coefficient matrix A (M,N)."""
        M, N = self.M, self.N
        # y_vec shape (L,), t_vec (L,)
        y_pows = torch.stack([y_vec**m for m in range(M)], dim=1)      # (L,M)
        t_pows = torch.stack([t_vec**n for n in range(N)], dim=1)      # (L,N)
        # F: sum_{m,n} a_{m,n} y^m t^n
        F = (y_pows.unsqueeze(2) * t_pows.unsqueeze(1) * A.unsqueeze(0)).sum(dim=(1,2))
        # dF/dy: sum_{m>=1,n} m a_{m,n} y^{m-1} t^n
        if M > 1:
            # build dPhi (L,M) for current y_vec
            dPhi_local = torch.zeros_like(y_pows)
            m_idx = torch.arange(M, device=self.device, dtype=self.dtype)
            if M > 1:
                dPhi_local[:, 1:] = m_idx[1:].unsqueeze(0) * y_pows[:, :-1]
            dF = (dPhi_local.unsqueeze(2) * t_pows.unsqueeze(1) * A.unsqueeze(0)).sum(dim=(1,2))
        else:
            dF = torch.zeros_like(y_vec)
        return F, dF

    def solve(self, func, y0, t_grid):
        """
        Solve ODE dy/dt = func(y,t) on t_grid by the ParODE method augmented
        with the PDE residual term. Returns y_pred (L,) and coefficient matrix A (M,N).
        """
        device, dtype = self.device, self.dtype
        y0 = torch.as_tensor(float(y0), device=device, dtype=dtype)
        t0, tf = float(t_grid[0].item()), float(t_grid[-1].item())

        print(f"Device: {device}, dtype: {dtype}")
        print(f"Basis: {self.M} x {self.N}, samples K={self.K}")

        # --- sampling collocation points and evaluate f ---
        ys_samples, ts_samples = self._sample_collocation(y0, t0, tf)
        # ensure func accepts tensors and returns tensor of shape (K,)
        f_k = func(ys_samples, ts_samples).to(device=device, dtype=dtype)

        # --- power matrices and PDE matrix P ---
        y_pows, t_pows, dPhi, dTpow = self._build_power_matrices(ys_samples, ts_samples)
        P, b_pde = self._assemble_pde_matrix(y_pows, t_pows, dPhi, dTpow, f_k)

        # Normal equations for PDE-fitting: P^T P a = P^T b
        PT_P = P.t() @ P          # (Nb, Nb)
        PT_b = P.t() @ b_pde      # (Nb,)

        # --- IC constraint: add weighted outer-product phi_ic phi_ic^T and rhs contribution
        phi_ic = self._assemble_ic_row(y0, t0)   # (Nb,)
        PT_P = PT_P + self.ic_weight * (phi_ic.unsqueeze(1) @ phi_ic.unsqueeze(0))
        PT_b = PT_b + self.ic_weight * (phi_ic * y0)

        # --- regularize and solve linear system ---
        Nb = self.M * self.N
        I = torch.eye(Nb, device=device, dtype=dtype)
        a = torch.linalg.solve(PT_P + self.reg_lambda * I, PT_b)  # (Nb,)
        A = a.view(self.M, self.N)

        # --- Newton fixed-point iteration (parallel over time grid) ---
        L = len(t_grid)
        # initialize y_batch with constant initial value y0
        y_batch = torch.full((L,), float(y0.item()), device=device, dtype=dtype)
        start_time = time.time()
        for it in range(self.max_newton_iters):
            Fv, dFdy = self._evaluate_F_and_dFdy(y_batch, t_grid, A)
            G = Fv - y_batch
            dG = dFdy - 1.0
            denom = torch.where(dG.abs() < 1e-12, torch.sign(dG) * 1e-12 + 1e-12, dG)
            delta = G / denom
            y_new = y_batch - self.damping * delta
            max_change = torch.max(torch.abs(y_new - y_batch)).item()
            rms_resid = torch.sqrt(torch.mean(G**2)).item()
            y_batch = y_new
            if (it % 10) == 0 or it == 0:
                print(f"Iter {it:3d}: max_change={max_change:.3e}, rms_resid={rms_resid:.3e}")
            if max_change < self.newton_tol:
                print(f"Converged at iter {it}, max_change={max_change:.3e}")
                break
        print(f"Newton loop time: {time.time()-start_time:.3f}s")

        return y_batch, A

    def compare_to_rk4(self, func, y0, t_grid, y_pred):
        """Compare ParODE solution to RK4 reference (same as in your original)."""
        def rk4_integrate(f, y0, t_grid):
            y = y0.clone()
            ys = [y.clone()]
            for i in range(len(t_grid) - 1):
                t = t_grid[i]
                dt = t_grid[i + 1] - t_grid[i]
                k1 = f(y, t)
                k2 = f(y + 0.5 * dt * k1, t + 0.5 * dt)
                k3 = f(y + 0.5 * dt * k2, t + 0.5 * dt)
                k4 = f(y + dt * k3, t + dt)
                y = y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
                ys.append(y.clone())
            return torch.stack(ys, dim=0)

        y_ref = rk4_integrate(func, torch.tensor(float(y0), device=self.device, dtype=self.dtype), t_grid)
        rel_err = torch.norm(y_ref - y_pred) / torch.norm(y_ref)
        dt = t_grid[1] - t_grid[0]
        dydt_fd = torch.empty_like(y_pred)
        dydt_fd[:-1] = (y_pred[1:] - y_pred[:-1]) / dt
        dydt_fd[-1] = dydt_fd[-2]
        residual = torch.norm(dydt_fd - func(y_pred, t_grid))
        print("\nDiagnostics vs RK4:")
        print(f"Relative L2 error: {rel_err.item():.3e}")
        print(f"Residual norm ||d/dt F - f||: {residual.item():.3e}")
        print("\n t     RK4_ref     F_fixedpt")
        for i in range(0, min(10, len(t_grid))):
            print(f"{t_grid[i].item():6.3f} {y_ref[i].item():16.10e} {y_pred[i].item():16.10e}")
        return rel_err.item(), residual.item()

# Example usage if run as script
if __name__ == "__main__":
    def f_rhs(y, t):
        # expect y and t are tensors (vectorized call)
        return -y**3 + torch.sin(t)
    
    solver = ParODE(M=10, N=10, K=200000)
    t_grid = torch.linspace(0.0, 2.0, 201, device=solver.device, dtype=solver.dtype)
    y0 = 0.5
    y_pred, A = solver.solve(f_rhs, y0, t_grid)
    solver.compare_to_rk4(f_rhs, y0, t_grid, y_pred)
    
    solver2 = ParODE2(M=10, N=10, K=200000, ic_weight=1e3, reg_lambda=1e-8)
    # t_grid = torch.linspace(0.0, 2.0, 201, device=solver.device, dtype=solver.dtype)
    # y0 = 0.5
    y_pred2, A2 = solver2.solve(f_rhs, y0, t_grid)
    solver2.compare_to_rk4(f_rhs, y0, t_grid, y_pred2)

