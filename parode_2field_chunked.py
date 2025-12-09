# parode_2field_chunked.py
# Chunked ParODE solver for two coupled fields U(u,v,t) and V(u,v,t)
# PDE residuals (to fit):
#   R_U = ∂_t U + f(u,v,t) ∂_u U + g(u,v,t) ∂_v U - f(u,v,t)  = 0
#   R_V = ∂_t V + f(u,v,t) ∂_u V + g(u,v,t) ∂_v V - g(u,v,t)  = 0
#
# Basis: monomials u^m v^n t^p with sizes M,N,P (m=0..M-1 etc)
# Coeffs: A_U (M,N,P) and A_V (M,N,P)
#
# The solver:
#  - assembles P^T P and P^T b for U and V in chunks over samples K
#  - solves linear systems for coefficients (regularized)
#  - iterates Newton per time point solving 2x2 linear systems to update u(t) and v(t)
#
# Author: adapted for Martin Veselsky (2025)
import math, time
import torch
import numpy as np

class ParODE2FieldChunked:
    def __init__(self,
                 M=6, N=6, P=6,        # basis sizes in u,v,t
                 K=200000,             # total collocation samples
                 ic_weight=1e3,
                 reg_lambda=1e-8,
                 max_newton_iters=200,
                 newton_tol=1e-10,
                 damping=0.8,
                 device=None,
                 dtype=torch.float64,
                 chunk_bytes_frac=0.40):
        self.M = int(M); self.N = int(N); self.P = int(P)
        self.K = int(K)
        self.ic_weight = float(ic_weight)
        self.reg_lambda = float(reg_lambda)
        self.max_newton_iters = int(max_newton_iters)
        self.newton_tol = float(newton_tol)
        self.damping = float(damping)
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.dtype = dtype
        self.chunk_bytes_frac = float(chunk_bytes_frac)

    def _nb(self):
        return self.M * self.N * self.P

    def suggest_K_chunk(self):
        """Heuristic suggestion for K_chunk given GPU memory and current Nb."""
        if not torch.cuda.is_available() or str(self.device).startswith("cpu"):
            return self.K
        try:
            props = torch.cuda.get_device_properties(self.device)
            total = props.total_memory
        except Exception:
            total = 16 * 1024**3
        budget = total * self.chunk_bytes_frac
        Nb = self._nb()
        dtype_bytes = torch.tensor([], dtype=self.dtype).element_size()
        alpha = 3.0   # safety factor for temporaries
        Kc = int(max(1, math.floor(budget / (alpha * dtype_bytes * Nb))))
        return max(1, min(Kc, self.K))

    # ----------------- sampling & powers (chunk) -----------------
    def _sample_chunk(self, u0, v0, t0, tf, Kc):
        """Return (u_chunk, v_chunk, t_chunk) all on device/dtype."""
        # deterministic-ish but we assume caller sets seed for reproducibility
        u_chunk = (float(u0) - 0.8) + 2.0 * 0.8 * torch.rand(Kc, device=self.device, dtype=self.dtype)
        v_chunk = (float(v0) - 0.8) + 2.0 * 0.8 * torch.rand(Kc, device=self.device, dtype=self.dtype)
        t_chunk = float(t0) + (float(tf) - float(t0)) * torch.rand(Kc, device=self.device, dtype=self.dtype)
        return u_chunk, v_chunk, t_chunk

    def _build_powers_chunk(self, u_c, v_c, t_c):
        """Return u_pows (Kc,M), v_pows (Kc,N), t_pows (Kc,P) and their derivatives dU,dV,dT."""
        Kc = u_c.shape[0]
        M, N, P = self.M, self.N, self.P
        u_pows = torch.stack([u_c**m for m in range(M)], dim=1)  # (Kc,M)
        v_pows = torch.stack([v_c**n for n in range(N)], dim=1)  # (Kc,N)
        t_pows = torch.stack([t_c**p for p in range(P)], dim=1)  # (Kc,P)
        dU = torch.zeros((Kc, M), device=self.device, dtype=self.dtype)
        if M > 1:
            m_idx = torch.arange(M, device=self.device, dtype=self.dtype)
            dU[:, 1:] = m_idx[1:].unsqueeze(0) * u_pows[:, :-1]
        dV = torch.zeros((Kc, N), device=self.device, dtype=self.dtype)
        if N > 1:
            n_idx = torch.arange(N, device=self.device, dtype=self.dtype)
            dV[:, 1:] = n_idx[1:].unsqueeze(0) * v_pows[:, :-1]
        dT = torch.zeros((Kc, P), device=self.device, dtype=self.dtype)
        if P > 1:
            p_idx = torch.arange(P, device=self.device, dtype=self.dtype)
            dT[:, 1:] = p_idx[1:].unsqueeze(0) * t_pows[:, :-1]
        return u_pows, v_pows, t_pows, dU, dV, dT

    def _assemble_P_chunk(self, u_pows, v_pows, t_pows, dU, dV, dT, f_k, g_k):
        """
        Build chunked design matrices for R_U and R_V.
        Ordering: flatten indices as (m, n, p) with m fastest or slowest? Use m-major then n then p:
           idx = (m * N + n) * P + p  -> consistent with A.view(M,N,P).
        We'll produce P_chunk_U and P_chunk_V of shape (Kc, Nb).
        Row (for a given sample k) entries are:
          basis_dt = u^m v^n * (p * t^{p-1})
          basis_du = (m * u^{m-1}) * v^n * t^p
          basis_dv = u^m * (n * v^{n-1}) * t^p
        P_row = basis_dt + f_k * basis_du + g_k * basis_dv
        """
        Kc = u_pows.shape[0]
        M, N, P = self.M, self.N, self.P
        # compute B_dt, B_du, B_dv as (Kc, M, N, P)
        B_dt = u_pows.unsqueeze(2).unsqueeze(3) * v_pows.unsqueeze(1).unsqueeze(3) * dT.unsqueeze(1).unsqueeze(2)
        B_du = dU.unsqueeze(2).unsqueeze(3) * v_pows.unsqueeze(1).unsqueeze(3) * t_pows.unsqueeze(1).unsqueeze(2)
        B_dv = u_pows.unsqueeze(2).unsqueeze(3) * dV.unsqueeze(1).unsqueeze(3) * t_pows.unsqueeze(1).unsqueeze(2)
        # combine (f_k,g_k shape (Kc,))
        fk = f_k.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        gk = g_k.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        P3_U = B_dt + fk * B_du + gk * B_dv  # (Kc,M,N,P)
        P3_V = B_dt + fk * (B_du) + gk * (B_dv)  # same form; residual b differs (see below)
        # flatten to (Kc, Nb)
        Nb = M * N * P
        P_chunk_U = P3_U.reshape(Kc, Nb)
        P_chunk_V = P3_V.reshape(Kc, Nb)
        return P_chunk_U, P_chunk_V

    # ----------------- solve (chunked assembly) -----------------
    def solve(self, func_f_g, u0, v0, t_grid, K_chunk=None, verbose=True):
        """
        func_f_g: callable accepting (u_tensor, v_tensor, t_tensor) shapes (Kc,) and returns tuple (f_k, g_k) both (Kc,)
        u0, v0: initial scalars
        t_grid: 1D tensor times (L,)
        K_chunk: chunk size for assembly (if None auto-suggested)
        Returns: u_batch (L,), v_batch (L,), A_U (M,N,P), A_V (M,N,P)
        """
        device, dtype = self.device, self.dtype
        u0 = torch.as_tensor(float(u0), device=device, dtype=dtype)
        v0 = torch.as_tensor(float(v0), device=device, dtype=dtype)
        t0, tf = float(t_grid[0].item()), float(t_grid[-1].item())
        Nb = self._nb()

        if K_chunk is None:
            K_chunk = self.suggest_K_chunk()
        K_chunk = max(1, min(int(K_chunk), self.K))

        if verbose:
            print(f"Device: {device}, dtype: {dtype}")
            print(f"Basis: {self.M} x {self.N} x {self.P} (Nb={Nb}), K total={self.K}, chunk={K_chunk}")

        # accumulators
        PT_P_U = torch.zeros((Nb, Nb), device=device, dtype=dtype)
        PT_b_U = torch.zeros((Nb,), device=device, dtype=dtype)
        PT_P_V = torch.zeros((Nb, Nb), device=device, dtype=dtype)
        PT_b_V = torch.zeros((Nb,), device=device, dtype=dtype)

        # reproducible sampling
        torch.manual_seed(0)
        left = self.K
        while left > 0:
            Kc = min(K_chunk, left)
            u_c, v_c, t_c = self._sample_chunk(u0, v0, t0, tf, Kc)
            f_k, g_k = func_f_g(u_c, v_c, t_c)
            f_k = f_k.to(device=device, dtype=dtype)
            g_k = g_k.to(device=device, dtype=dtype)
            u_pows, v_pows, t_pows, dU, dV, dT = self._build_powers_chunk(u_c, v_c, t_c)
            P_chunk_U, P_chunk_V = self._assemble_P_chunk(u_pows, v_pows, t_pows, dU, dV, dT, f_k, g_k)
            # b vectors: for R_U equation RHS is f_k; for R_V equation RHS is g_k
            PT_P_U += P_chunk_U.t().matmul(P_chunk_U)
            PT_b_U += P_chunk_U.t().matmul(f_k)
            PT_P_V += P_chunk_V.t().matmul(P_chunk_V)
            PT_b_V += P_chunk_V.t().matmul(g_k)
            left -= Kc
            if verbose:
                print(f"  chunk processed Kc={Kc}, remaining={left}")
            # free chunk locals
            del u_c, v_c, t_c, f_k, g_k, u_pows, v_pows, t_pows, dU, dV, dT, P_chunk_U, P_chunk_V
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # add IC constraint (enforce U(u0,v0,t0)=u0, V(u0,v0,t0)=v0)
        phi_ic = torch.empty((Nb,), device=device, dtype=dtype)
        col = 0
        for m in range(self.M):
            u_pow = (u0**m)
            for n in range(self.N):
                v_pow = (v0**n)
                for p in range(self.P):
                    t_pow = (t0**p)
                    phi_ic[col] = u_pow * v_pow * t_pow
                    col += 1
        PT_P_U = PT_P_U + self.ic_weight * (phi_ic.unsqueeze(1) @ phi_ic.unsqueeze(0))
        PT_b_U = PT_b_U + self.ic_weight * (phi_ic * u0)
        PT_P_V = PT_P_V + self.ic_weight * (phi_ic.unsqueeze(1) @ phi_ic.unsqueeze(0))
        PT_b_V = PT_b_V + self.ic_weight * (phi_ic * v0)

        # regularize and solve linear systems for a_U and a_V
        I = torch.eye(Nb, device=device, dtype=dtype)
        a_U = torch.linalg.solve(PT_P_U + self.reg_lambda * I, PT_b_U)
        a_V = torch.linalg.solve(PT_P_V + self.reg_lambda * I, PT_b_V)
        A_U = a_U.view(self.M, self.N, self.P)
        A_V = a_V.view(self.M, self.N, self.P)

        # ---------------- Newton coupled iteration over t_grid ----------------
        L = len(t_grid)
        u_batch = torch.full((L,), float(u0.item()), device=device, dtype=dtype)
        v_batch = torch.full((L,), float(v0.item()), device=device, dtype=dtype)
        start = time.time()
        # precompute t-powers for t_grid (small)
        t_pows_full = torch.stack([t_grid**p for p in range(self.P)], dim=1).to(device=device, dtype=dtype)  # (L,P)
        for it in range(self.max_newton_iters):
            # compute F_U, F_V and partials wrt u,v (vectorized across L)
            u_pows_full = torch.stack([u_batch**m for m in range(self.M)], dim=1)  # (L,M)
            v_pows_full = torch.stack([v_batch**n for n in range(self.N)], dim=1)  # (L,N)
            # F_U (L,) = sum_{mnp} A_U[m,n,p] * u^m * v^n * t^p
            F_U = (u_pows_full.unsqueeze(2).unsqueeze(3) * v_pows_full.unsqueeze(1).unsqueeze(3)
                   * t_pows_full.unsqueeze(1).unsqueeze(2) * A_U.unsqueeze(0)).sum(dim=(1,2,3))
            F_V = (u_pows_full.unsqueeze(2).unsqueeze(3) * v_pows_full.unsqueeze(1).unsqueeze(3)
                   * t_pows_full.unsqueeze(1).unsqueeze(2) * A_V.unsqueeze(0)).sum(dim=(1,2,3))
            # partials: dF_U/du = sum m * A_U * u^{m-1} v^n t^p
            if self.M > 1:
                dU_local = torch.zeros_like(u_pows_full)
                m_idx = torch.arange(self.M, device=device, dtype=dtype)
                dU_local[:, 1:] = m_idx[1:].unsqueeze(0) * u_pows_full[:, :-1]
                dF_U_du = (dU_local.unsqueeze(2).unsqueeze(3) * v_pows_full.unsqueeze(1).unsqueeze(3)
                           * t_pows_full.unsqueeze(1).unsqueeze(2) * A_U.unsqueeze(0)).sum(dim=(1,2,3))
            else:
                dF_U_du = torch.zeros((L,), device=device, dtype=dtype)
            if self.N > 1:
                dV_local = torch.zeros_like(v_pows_full)
                n_idx = torch.arange(self.N, device=device, dtype=dtype)
                dV_local[:, 1:] = n_idx[1:].unsqueeze(0) * v_pows_full[:, :-1]
                dF_U_dv = (u_pows_full.unsqueeze(2).unsqueeze(3) * dV_local.unsqueeze(1).unsqueeze(3)
                           * t_pows_full.unsqueeze(1).unsqueeze(2) * A_U.unsqueeze(0)).sum(dim=(1,2,3))
            else:
                dF_U_dv = torch.zeros((L,), device=device, dtype=dtype)
            # same for V
            if self.M > 1:
                dF_V_du = (dU_local.unsqueeze(2).unsqueeze(3) * v_pows_full.unsqueeze(1).unsqueeze(3)
                           * t_pows_full.unsqueeze(1).unsqueeze(2) * A_V.unsqueeze(0)).sum(dim=(1,2,3))
            else:
                dF_V_du = torch.zeros((L,), device=device, dtype=dtype)
            if self.N > 1:
                dF_V_dv = (u_pows_full.unsqueeze(2).unsqueeze(3) * dV_local.unsqueeze(1).unsqueeze(3)
                           * t_pows_full.unsqueeze(1).unsqueeze(2) * A_V.unsqueeze(0)).sum(dim=(1,2,3))
            else:
                dF_V_dv = torch.zeros((L,), device=device, dtype=dtype)

            # Build residual vectors and jacobians per time point:
            # G = [F_U - u; F_V - v], Jacobian J = [[dF_U/du -1, dF_U/dv],
            #                                       [dF_V/du,    dF_V/dv -1]]
            G1 = F_U - u_batch
            G2 = F_V - v_batch
            a11 = dF_U_du - 1.0
            a12 = dF_U_dv
            a21 = dF_V_du
            a22 = dF_V_dv - 1.0
            # Solve 2x2 linear system J * delta = G for each time point
            det = a11 * a22 - a12 * a21
            # regularize small determinants
            small = det.abs() < 1e-14
            det_reg = det.clone()
            det_reg[small] = det_reg[small] + 1e-14 * torch.sign(det_reg[small] + 1e-30)
            inv11 =  a22 / det_reg
            inv12 = -a12 / det_reg
            inv21 = -a21 / det_reg
            inv22 =  a11 / det_reg
            delta_u = inv11 * G1 + inv12 * G2
            delta_v = inv21 * G1 + inv22 * G2
            # update
            u_new = u_batch - self.damping * delta_u
            v_new = v_batch - self.damping * delta_v
            max_change = max(torch.max(torch.abs(u_new - u_batch)).item(),
                             torch.max(torch.abs(v_new - v_batch)).item())
            rms_resid = math.sqrt(float(torch.mean(G1**2 + G2**2).item()))
            u_batch = u_new
            v_batch = v_new
            if verbose and ((it % 10) == 0 or it == 0):
                print(f"Iter {it:3d}: max_change={max_change:.3e}, rms_resid={rms_resid:.3e}")
            if max_change < self.newton_tol:
                if verbose:
                    print(f"Converged at iter {it}, max_change={max_change:.3e}")
                break
        if verbose:
            print(f"Newton loop time: {time.time()-start:.3f}s")
        return u_batch, v_batch, A_U, A_V

    def compare_to_reference(self, func_f_g, u0, v0, t_grid, u_pred, v_pred):
        """
        Compare to a sequential reference integrator (RK4) for coupled system:
           du/dt = f(u,v,t), dv/dt = g(u,v,t)
        func_f_g: callable returning (f,g) given (u,v,t) scalars or vectorized over L
        """
        device, dtype = self.device, self.dtype
        def rk4_system(func, u0, v0, t_grid):
            # sequential scalar RK4 for small L (CPU or GPU)
            u = torch.tensor(float(u0), device=device, dtype=dtype)
            v = torch.tensor(float(v0), device=device, dtype=dtype)
            us = [u.clone()]
            vs = [v.clone()]
            for i in range(len(t_grid)-1):
                t = t_grid[i]
                dt = t_grid[i+1] - t
                f1, g1 = func(u, v, t)
                f2, g2 = func(u + 0.5*dt*f1, v + 0.5*dt*g1, t + 0.5*dt)
                f3, g3 = func(u + 0.5*dt*f2, v + 0.5*dt*g2, t + 0.5*dt)
                f4, g4 = func(u + dt*f3, v + dt*g3, t + dt)
                u = u + dt*(f1 + 2*f2 + 2*f3 + f4)/6.0
                v = v + dt*(g1 + 2*g2 + 2*g3 + g4)/6.0
                us.append(u.clone()); vs.append(v.clone())
            return torch.stack(us, dim=0), torch.stack(vs, dim=0)
        u_ref, v_ref = rk4_system(func_f_g, u0, v0, t_grid)
        np.savetxt("rk4_compare.out", np.c_[u_pred.cpu(),v_pred.cpu(),u_ref.cpu(),v_ref.cpu()])
        rel_err_u = torch.norm(u_ref - u_pred) / torch.norm(u_ref)
        rel_err_v = torch.norm(v_ref - v_pred) / torch.norm(v_ref)
        print(f"Rel err u: {rel_err_u.item():.3e}, v: {rel_err_v.item():.3e}")
        return rel_err_u.item(), rel_err_v.item()

