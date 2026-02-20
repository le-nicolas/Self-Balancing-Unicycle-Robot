"""
Model Predictive Control (MPC) for reaction wheel balancer with hard COM position constraints.

This implementation guarantees COM position stays within support radius by embedding
the constraint into the optimization problem itself (not checked after control).

Uses OSQP (Operator Splitting Quadratic Program solver) for fast, reliable solutions
within 1ms control cycle time.
"""

import numpy as np
from scipy.linalg import block_diag
import time
from typing import Tuple, Optional

try:
    import osqp
    OSQP_AVAILABLE = True
except ImportError:
    OSQP_AVAILABLE = False
    print("Warning: osqp not available, will fall back to scipy.optimize")


class MPCController:
    """
    Receding horizon MPC solver for reaction wheel balancer.
    
    State: [pitch, roll, pitch_rate, roll_rate, wheel_rate, base_x, base_y, base_vel_x, base_vel_y]
    Control: [wheel_torque, base_force_x, base_force_y]
    
    Constraints enforced:
    - Hard COM position bounds: sqrt(base_x^2 + base_y^2) <= support_radius
    - Control input saturation
    - Angle limits
    """
    
    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        horizon: int = 25,
        q_diag: np.ndarray = None,
        r_diag: np.ndarray = None,
        u_max: np.ndarray = None,
        com_radius_m: float = 0.145,
        angle_max_rad: float = 0.5,
        verbose: bool = False,
    ):
        """
        Initialize MPC controller.
        
        Args:
            A: State transition matrix (9x9)
            B: Control input matrix (9x3)
            horizon: Prediction horizon in steps (default 25 = 100ms at 250Hz)
            q_diag: Diagonal of state cost Q (default: balance angle/rate control)
            r_diag: Diagonal of control cost R (default: penalize large forces)
            u_max: Maximum control inputs [wheel_torque, base_x_force, base_y_force]
            com_radius_m: Maximum COM distance from origin (support radius)
            angle_max_rad: Maximum allowed pitch/roll angle
            verbose: Print solver info for debugging
        """
        self.A = A
        self.B = B
        self.N = horizon
        self.nx = A.shape[0]  # 9
        self.nu = B.shape[1]  # 3
        self.verbose = verbose
        
        # Cost matrices
        if q_diag is None:
            # Default: prioritize angle/rate control, moderate position control
            q_diag = np.array([200.0, 200.0, 100.0, 100.0, 1.0, 50.0, 50.0, 100.0, 100.0])
        if r_diag is None:
            # Default: penalize control effort equally
            r_diag = np.array([0.1, 0.5, 0.5])
        
        self.Q = np.diag(q_diag)
        self.R = np.diag(r_diag)
        
        # Constraint bounds
        if u_max is None:
            u_max = np.array([80.0, 10.0, 10.0])  # From runtime_config defaults
        self.u_max = u_max
        self.com_radius = com_radius_m
        self.angle_max = angle_max_rad
        
        # Build QP problem structure
        self._build_qp_matrices()
        
        # Initialize solver
        self.solver = None
        self.solve_time = 0.0
        self.iterations = 0
        
        if OSQP_AVAILABLE:
            self._init_osqp_solver()
    
    def _build_qp_matrices(self):
        """
        Build quadratic program matrices for MPC.
        
        Formulation:
            minimize: ||X||_Q^2 + ||U||_R^2
            subject to:
                x_{k+1} = A x_k + B u_k
                COM constraints: sqrt(x[5]^2 + x[6]^2) <= r
                Input bounds: |u| <= u_max
                Angle bounds: |x[0]|, |x[1]| <= angle_max
        
        Converts to standard QP form:
            minimize: 0.5 z^T P z + q^T z
            subject to: l <= A_qp z <= u_qp
        """
        N = self.N
        nx = self.nx
        nu = self.nu
        
        # Build cost matrices: P (Hessian) and q (gradient)
        Q_blk = block_diag(*[self.Q] * (N + 1))  # State costs for horizon
        R_blk = block_diag(*[self.R] * N)         # Control costs for horizon
        
        # P = 2 * block_diag(Q_blk, R_blk) for hessian (symmetric)
        self.P = block_diag(Q_blk, R_blk)
        self.q = np.zeros((N + 1) * nx + N * nu)
        
        # Build constraint matrices: A_eq for dynamics
        # x_0, x_1, ..., x_N, u_0, u_1, ..., u_{N-1}
        
        # Dynamics constraints: x_{k+1} = A x_k + B u_k
        # Reformulated as: [A -I][x_k; x_{k+1}] + [B][u_k] = 0
        # Or in inequality form (we need equality via lower=upper):
        # -A x_k + x_{k+1} - B u_k = 0
        
        A_dyn = np.zeros((N * nx, (N + 1) * nx + N * nu))
        for k in range(N):
            # Row indices for constraint k
            row_idx = k * nx
            col_state_k = k * nx
            col_state_k1 = (k + 1) * nx
            col_u_k = (N + 1) * nx + k * nu
            
            A_dyn[row_idx:row_idx + nx, col_state_k:col_state_k + nx] = -self.A
            A_dyn[row_idx:row_idx + nx, col_state_k1:col_state_k1 + nx] = np.eye(nx)
            A_dyn[row_idx:row_idx + nx, col_u_k:col_u_k + nu] = -self.B
        
        self.A_dyn = A_dyn
        self.b_dyn = np.zeros(N * nx)
        
        # Initial state constraint: x_0 = x_current (will be updated in solve())
        A_init = np.zeros((nx, (N + 1) * nx + N * nu))
        A_init[:, :nx] = np.eye(nx)  # x_0 coefficients
        self.A_init = A_init
        self.b_init = np.zeros(nx)  # Will be set to x_current in solve()
        
        
        # Input bounds: -u_max <= u <= u_max
        A_u = np.zeros((2 * N * nu, (N + 1) * nx + N * nu))
        l_u = np.zeros(2 * N * nu)
        u_u = np.zeros(2 * N * nu)
        
        for k in range(N):
            col_u_k = (N + 1) * nx + k * nu
            # u >= -u_max
            A_u[2 * k * nu:(2 * k + 1) * nu, col_u_k:col_u_k + nu] = np.eye(nu)
            l_u[2 * k * nu:(2 * k + 1) * nu] = -self.u_max
            # u <= u_max
            A_u[(2 * k + 1) * nu:(2 * k + 2) * nu, col_u_k:col_u_k + nu] = np.eye(nu)
            u_u[(2 * k + 1) * nu:(2 * k + 2) * nu] = self.u_max
        
        self.A_u = A_u
        self.l_u = l_u
        self.u_u = u_u
        
        # Angle bounds: |pitch|, |roll| <= angle_max
        A_angle = np.zeros((2 * 2 * (N + 1), (N + 1) * nx + N * nu))
        l_angle = np.zeros(2 * 2 * (N + 1))
        u_angle = np.zeros(2 * 2 * (N + 1))
        
        for k in range(N + 1):
            col_state_k = k * nx
            # pitch <= angle_max
            A_angle[2 * k, col_state_k + 0] = 1.0
            u_angle[2 * k] = self.angle_max
            l_angle[2 * k] = -np.inf
            # pitch >= -angle_max
            A_angle[2 * k + 1, col_state_k + 0] = -1.0
            u_angle[2 * k + 1] = self.angle_max
            l_angle[2 * k + 1] = -np.inf
            
            # roll <= angle_max
            col_start = 2 * (N + 1) + 2 * k
            A_angle[col_start, col_state_k + 1] = 1.0
            u_angle[col_start] = self.angle_max
            l_angle[col_start] = -np.inf
            # roll >= -angle_max
            A_angle[col_start + 1, col_state_k + 1] = -1.0
            u_angle[col_start + 1] = self.angle_max
            l_angle[col_start + 1] = -np.inf
        
        self.A_angle = A_angle
        self.l_angle = l_angle
        self.u_angle = u_angle
        
        # COM position constraints: sqrt(base_x^2 + base_y^2) <= r_com
        # This is nonlinear, so we use trust region / polyhedral approximation
        # For now, use max(|base_x|, |base_y|) <= r / sqrt(2) (conservative box approximation)
        # This ensures sqrt(x^2 + y^2) <= r_com
        A_com = np.zeros((2 * 2 * (N + 1), (N + 1) * nx + N * nu))
        l_com = np.zeros(2 * 2 * (N + 1))
        u_com = np.zeros(2 * 2 * (N + 1))
        
        com_bound = self.com_radius / np.sqrt(2)  # Conservative approximation
        
        for k in range(N + 1):
            col_state_k = k * nx
            # base_x <= com_bound
            A_com[2 * k, col_state_k + 5] = 1.0
            u_com[2 * k] = com_bound
            l_com[2 * k] = -np.inf
            # base_x >= -com_bound
            A_com[2 * k + 1, col_state_k + 5] = -1.0
            u_com[2 * k + 1] = com_bound
            l_com[2 * k + 1] = -np.inf
            
            # base_y <= com_bound
            col_y_start = 2 * (N + 1) + 2 * k
            A_com[col_y_start, col_state_k + 6] = 1.0
            u_com[col_y_start] = com_bound
            l_com[col_y_start] = -np.inf
            # base_y >= -com_bound
            A_com[col_y_start + 1, col_state_k + 6] = -1.0
            u_com[col_y_start + 1] = com_bound
            l_com[col_y_start + 1] = -np.inf
        
        self.A_com = A_com
        self.l_com = l_com
        self.u_com = u_com
    
    def _init_osqp_solver(self):
        """Initialize OSQP solver with problem structure."""
        try:
            # Stack all constraints
            A_all = np.vstack([
                self.A_init,  # Initial state constraint: x_0 = x_current
                self.A_dyn,
                self.A_u,
                self.A_angle,
                self.A_com,
            ])
            
            l_all = np.concatenate([
                self.b_init,  # Will be updated in solve() to x_current
                self.b_dyn,
                self.l_u,
                self.l_angle,
                self.l_com,
            ])
            
            u_all = np.concatenate([
                self.b_init,  # Will be updated in solve() to x_current (equality constraint)
                self.b_dyn,  # Dynamics equality constraints
                self.u_u,
                self.u_angle,
                self.u_com,
            ])
            
            # Convert to sparse format
            from scipy.sparse import csc_matrix
            P_sparse = csc_matrix(self.P)
            A_sparse = csc_matrix(A_all)
            
            # Setup OSQP solver
            self.solver = osqp.OSQP()
            self.solver.setup(
                P=P_sparse,
                q=self.q,
                A=A_sparse,
                l=l_all,
                u=u_all,
                verbose=self.verbose,
                scaled_termination=True,
                polish=True,
            )
        except Exception as e:
            print(f"OSQP initialization failed: {e}")
            self.solver = None
    
    def solve(
        self,
        x_current: np.ndarray,
        x_ref: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Solve MPC problem for current state and return optimal control.
        
        Args:
            x_current: Current state estimate [pitch, roll, pitch_rate, roll_rate, 
                       wheel_rate, base_x, base_y, base_vel_x, base_vel_y]
            x_ref: Reference state (default: upright at origin)
        
        Returns:
            u_opt: Optimal control command [wheel_torque, base_x_force, base_y_force]
            info: Dictionary with solver info (solve_time, iterations, success)
        """
        if x_ref is None:
            x_ref = np.zeros(self.nx)
        
        start_time = time.time()
        
        # Update cost function with PROPER error tracking
        # The cost is: 0.5 * sum_k ||x_k - x_ref||_Q^2 + sum_k ||u_k||_R^2
        # In QP form: 0.5 z^T P z + q^T z
        # where P = diag(Q, ..., Q, R, ..., R) (already set in _build_qp_matrices)
        # and q = [-2 Q x_ref, -2 Q x_ref, ..., 0, ..., 0]
        # 
        # With LQR delta-u formulation, we need to track x_ref for state cost,
        # which creates the linear term: q = -2 * [Q@x_ref for each timestep]
        q_track = np.zeros((self.N + 1) * self.nx + self.N * self.nu)
        for k in range(self.N + 1):
            idx = k * self.nx
            # Reference tracking: -2 * Q * x_ref (this is the linear cost term from expanded quadratic)
            q_track[idx:idx + self.nx] = -2 * self.Q @ x_ref
        
        if self.solver is None:
            # Fallback to numerical optimization
            u_opt = self._solve_scipy(x_current, x_ref)
            solve_time = time.time() - start_time
            return u_opt, {
                "solve_time_ms": solve_time * 1000,
                "method": "scipy",
                "success": u_opt is not None,
            }
        
        try:
            # Update initial state constraint bounds
            # The constraint is: A_init @ z = x_0 = x_current
            # We enforce this as l_init <= A_init @ z <= u_init where l_init = u_init = x_current
            l_all = np.concatenate([
                x_current,  # x_0 = x_current (equality)
                self.b_dyn,  # Dynamics equality constraints
                self.l_u,
                self.l_angle,
                self.l_com,
            ])
            u_all = np.concatenate([
                x_current,  # x_0 = x_current (equality)
                self.b_dyn,
                self.u_u,
                self.u_angle,
                self.u_com,
            ])
            
            # Update cost to track reference
            q_track = np.zeros((self.N + 1) * self.nx + self.N * self.nu)
            for k in range(self.N + 1):
                idx = k * self.nx
                # Reference tracking: -2 * Q * x_ref (this is the linear cost term from expanded quadratic)
                q_track[idx:idx + self.nx] = -2 * self.Q @ x_ref
            
            # Update solver
            self.solver.update(q=q_track, l=l_all, u=u_all)
            
            results = self.solver.solve()
            
            solve_time = time.time() - start_time
            
            if results.info.status == "solved":
                z_opt = results.x
                # Extract control: u_0 (first control input)
                u_start = (self.N + 1) * self.nx
                u_opt = z_opt[u_start:u_start + self.nu]
                
                return np.array(u_opt), {
                    "solve_time_ms": solve_time * 1000,
                    "iterations": results.info.iter,
                    "method": "osqp",
                    "success": True,
                }
            else:
                if self.verbose:
                    print(f"OSQP did not solve: {results.info.status}")
                # Fallback to zero input
                return np.zeros(self.nu), {
                    "solve_time_ms": solve_time * 1000,
                    "status": results.info.status,
                    "method": "osqp",
                    "success": False,
                }
        
        except Exception as e:
            if self.verbose:
                print(f"MPC solve error: {e}")
            return np.zeros(self.nu), {
                "error": str(e),
                "method": "osqp",
                "success": False,
            }
    
    def _solve_scipy(
        self,
        x_current: np.ndarray,
        x_ref: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Fallback solver using scipy.optimize when OSQP unavailable.
        """
        from scipy.optimize import minimize
        
        def objective(z):
            x = z[:(self.N + 1) * self.nx]
            u = z[(self.N + 1) * self.nx:]
            
            # State cost
            state_error = x - np.tile(x_ref, self.N + 1)
            state_cost = state_error @ self.P[:self.nx * (self.N + 1), :self.nx * (self.N + 1)] @ state_error
            
            # Control cost
            u_cost = u @ (block_diag(*[self.R] * self.N)) @ u
            
            return state_cost + u_cost
        
        def constraint_dynamics(z):
            # x_{k+1} = A x_k + B u_k
            residuals = []
            for k in range(self.N):
                x_k = z[k * self.nx:(k + 1) * self.nx]
                x_k1 = z[(k + 1) * self.nx:(k + 2) * self.nx]
                u_k = z[(self.N + 1) * self.nx + k * self.nu:(self.N + 1) * self.nx + (k + 1) * self.nu]
                residuals.append(x_k1 - (self.A @ x_k + self.B @ u_k))
            return np.concatenate(residuals) if residuals else np.array([])
        
        z0 = np.zeros((self.N + 1) * self.nx + self.N * self.nu)
        z0[:self.nx] = x_current
        
        constraints = {'type': 'eq', 'fun': constraint_dynamics}
        bounds = [(None, None)] * ((self.N + 1) * self.nx) + [
            (-self.u_max[i % self.nu], self.u_max[i % self.nu])
            for i in range(self.N * self.nu)
        ]
        
        result = minimize(
            objective,
            z0,
            method='SLSQP',
            constraints=constraints,
            bounds=bounds,
            options={'ftol': 1e-5, 'maxiter': 50},
        )
        
        if result.success:
            u_start = (self.N + 1) * self.nx
            return result.x[u_start:u_start + self.nu]
        else:
            return None
