class Manifold:
    """Abstract base class for Riemannian manifolds.

    Does not inherit from nn.Module -- concrete subclasses that need
    learnable parameters should use multiple inheritance
    (e.g., class PoincareBall(nn.Module, Manifold)).
    """

    def proj(self, x):
        """Project point x onto the manifold (numerical safety)."""
        raise NotImplementedError

    def proj_tan(self, u, x):
        """Project vector u onto the tangent space at x."""
        raise NotImplementedError

    def expmap(self, u, x):
        """Exponential map at x: T_x M -> M."""
        raise NotImplementedError

    def logmap(self, y, x):
        """Logarithmic map at x: M -> T_x M."""
        raise NotImplementedError

    def expmap0(self, u):
        """Exponential map at the origin."""
        raise NotImplementedError

    def logmap0(self, x):
        """Logarithmic map at the origin."""
        raise NotImplementedError

    def ptransp(self, x, y, u):
        """Parallel transport of u from T_x M to T_y M."""
        raise NotImplementedError

    def ptransp0(self, y, u):
        """Parallel transport from origin to y."""
        raise NotImplementedError

    def dist(self, x, y):
        """Geodesic distance between x and y."""
        raise NotImplementedError

    def mobius_add(self, x, y):
        """Mobius addition."""
        raise NotImplementedError

    def mobius_matvec(self, m, x):
        """Mobius matrix-vector multiplication."""
        raise NotImplementedError

    @property
    def name(self):
        raise NotImplementedError
