import torch
from parode_class import ParODE

# ---------------- Example usage ----------------
if __name__ == "__main__":
    def f_rhs(y, t):
        return -y**3 + torch.sin(t)

    solver = ParODE(M=10, N=10, K=200000)
    t_grid = torch.linspace(0.0, 2.0, 201, device=solver.device, dtype=solver.dtype)
    y0 = 0.5
    y_pred, A = solver.solve(f_rhs, y0, t_grid)
    solver.compare_to_rk4(f_rhs, y0, t_grid, y_pred)
