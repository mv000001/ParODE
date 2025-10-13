# parode_class.py
# GPU-parallel fixed-point ODE solver using polynomial basis collocation
# Author: Martin Veselsky (2025)
# Equation form: dy/dt = f(y,t), solved via polynomial expansion F(y,t)=Σ a_mn y^m t^n
# The solution satisfies F(y0,t0)=y0 and F(y,t)-y=0 by Newton iteration.

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
            print(f"{t_grid[i].item():6.3f} {y_ref[i].item():12.6e} {y_pred[i].item():12.6e}")
    
        return rel_err.item(), residual.item()

## ---------------- Example usage ----------------
#if __name__ == "__main__":
#    def f_rhs(y, t):
#        return -y**3 + torch.sin(t)
#
#    solver = ParODE(M=10, N=10, K=200000)
#    t_grid = torch.linspace(0.0, 0.1, 201, device=solver.device, dtype=solver.dtype)
#    y0 = 0.5
#    y_pred, A = solver.solve(f_rhs, y0, t_grid)
#    solver.compare_to_rk4(f_rhs, y0, t_grid, y_pred)
