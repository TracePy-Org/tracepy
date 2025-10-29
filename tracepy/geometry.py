import numpy as np

from .utils import gen_rot
from .exceptions import NotOnSurfaceError, InvalidGeometry
from .index import glass_index

from typing import Dict, List, Tuple, Union, Optional

class geometry:
    """Class for the different surfaces in an optical system.

    Attributes
    ----------
    P : list of 3 floats/ints or just float/int
        Position of surface in the lab frame. If only one float or int is given
        then it is considered the position in the Z direction.
    D (optional): list of 3 floats/ints
        Rotation angles alpha, beta, gamma for a surface.
        Defaults to no rotation -> [0., 0., 0.]
    action : str
        Interaction of rays with the specified surface. Either reflection,
        refraction, or stop (do nothing).
    Diam : float/int
        Outer diameter of the specified surface.
    diam : float/int
        Inner diameter of the specified surface.
    N (optional): float/int
        Index of refraction of the material that the rays propagate INTO
        (meaning on the image side). If N is not given then it defaults
        to a value of 1 for air. A constant value can be specified such as 1.72.
    kappa (optional): float/int
        Defines the type of conic that the surface models.
        If kappa is None then the surface is planar.
        kappa < 0 -> hyperboloid
        kappa = 0 -> paraboloid
        0 < kappa < 1 -> helioid of revolution about major axis
        kappa = 1 -> hemisphere
        kappa > 1 -> helioid of revolution about minor axis
    c (optional): float/int
        Vertex curvature of the surface.
        If c is 0 then the surface is planar.
    name (optional): str
        Name of the surface, used for optimization
    R (generated): np.array((3,3))
        Rotation matrix for the surface from rotation angles D.

    """

    def __init__(self, params: Dict):
        P = params.get('P')
        # Allow on axis integer for P.
        if isinstance(P, float) or isinstance(P, int):
            P = np.array([0., 0., P])
        elif isinstance(P, List):
            P = np.array(P)
        elif not isinstance(P, np.ndarray):
            raise InvalidGeometry()
        self.P: np.ndarray = P
        self.D: np.ndarray = np.array(params.get('D', [0., 0., 0.]))
        self.action: str = params['action']
        self.Diam: Union[float, int] = params['Diam']
        self.N: Union[float, int] = params.get('N', 1.)
        self.kappa: Optional[Union[float, int]] = params.get('kappa')
        self.diam: Union[float, int] = params.get('diam', 0.)
        self.c: Union[float, int] = params.get('c', 0.)
        self.name: str = params.get('name', None)
        self.R: np.ndarray = gen_rot(self.D)
        if params.get('glass'):
            self.glass = glass_index(params.get('glass'))
        self.check_params()

    def check_params(self) -> None:
        """Check that required parameters are given and update needed parameters.

        Summary
        -------
            If c != 0 (in the case of a conic) then kappa must be specified,
            and if kappa is greater than 0 then the value of c is redundant
            by boundary conditions of the conic equation. Lastly, if c == 0 in
            the case of a planar surface the None value of kappa needs to be set
            to a dummy value to avoid exceptions in calculating the conic equation.
            Note that this does not affect the calculation since c is 0.

        """
        if self.c != 0:
            if self.kappa is None:
                raise Exception("Specify a kappa for this conic.")
            elif self.kappa > 0:
                print("Warning: Specified c value is not used when kappa>0")
                self.c = np.sqrt(1 / (self.kappa * (self.Diam / 2.)**2))
        elif self.c == 0 and self.kappa is None:
            # Used for planes, does not affect calculations.
            self.kappa = 1.

    def get_surface(self, point: np.ndarray) -> Tuple[float, List[float]]:
        """ Returns the function and derivative of a surface for a point. """
        return self.conics(point)

    def get_surface_plot(self, points: np.ndarray) -> np.ndarray:
        """ Returns the function value for an array of points. """
        return self.conics_plot(points)

    def get_surface_vector(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns function values and derivative vectors for an array of points.

        Parameters
        ----------
        points : np.ndarray of shape (N,3)
            Array of points.
        Returns
        -------
        func : np.ndarray of shape (N,)
            Function values.
        deriv : np.ndarray of shape (N,3)
            Derivative vectors.
        """
        return self.conics_vector(points)

    def conics(self, point: np.ndarray) -> Tuple[float, List[float]]:
        """Returns function value and derivative list for conics and sphere surfaces.

        Note
        ----
        This differs from the conics_plot method because we are returning the function
        value from the conic at the point X,Y,Z rather than the Z value for a given X, Y
        pair. A surface is defined by the equation F(X, Y, Z) = 0, so if the ray
        intersects with the surface this function will return a function value of
        0.

        Parameters
        ----------
        point : np.array of length 3
            X, Y, and Z values for the point.

        Returns
        -------
        function : float
            Function value of the surface. A value of 0 corresponds to the ray intersecting
            the surface.
        derivative : list of length 3
            Derivative of the surface at the point given.
        """
        X, Y, Z = point
        rho_sq = X**2 + Y**2
        rho = np.sqrt(rho_sq)
        if rho > self.Diam / 2. or rho < self.diam / 2.:
            raise NotOnSurfaceError()
        if self.kappa is None:
            raise ValueError("kappa must not be None for conic calculations")
        sqrt_term = (1 - self.kappa * self.c**2 * rho_sq)**0.5
        function = Z - self.c * rho_sq / (1 + sqrt_term)
        E = self.c / sqrt_term
        derivative = [-X * E, -Y * E, 1.]
        return function, derivative

    def conics_vector(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns function values and derivative vectors for conics and sphere surfaces.
        
        Vectorized version for an array of points.
        
        Parameters
        ----------
        points : np.ndarray of shape (N,3)
            Array of points (X, Y, Z).
        
        Returns
        -------
        func : np.ndarray of shape (N,)
            Function values for each point.
        deriv : np.ndarray of shape (N,3)
            Derivative vectors for each point.
        """
        X = points[:, 0]
        Y = points[:, 1]
        Z = points[:, 2]
        rho = np.sqrt(X**2 + Y**2)
        # Initialize outputs
        func = np.full(points.shape[0], np.nan)
        deriv = np.full((points.shape[0], 3), np.nan)
        # Determine valid indices based on aperture
        valid = (rho <= self.Diam / 2.) & (rho >= self.diam / 2.)
        if self.kappa is None:
            raise ValueError("kappa must not be None for conic calculations")
        # Calculate for valid points
        rho_valid = rho[valid]
        sqrt_term = np.sqrt(1 - self.kappa * self.c**2 * rho_valid**2)
        denom = 1 + sqrt_term
        func[valid] = Z[valid] - self.c * rho_valid**2 / denom
        E = self.c / sqrt_term
        deriv[valid, 0] = -X[valid] * E
        deriv[valid, 1] = -Y[valid] * E
        deriv[valid, 2] = 1.
        return func, deriv

    def conics_plot(self, point: np.ndarray) -> np.ndarray:
        """Returns Z values for an array of points for plotting conics.

        Parameters
        ----------
        point : 2d np.array with shape (n, 3)
            X, Y, and Z values for each point (for each row).

        Returns
        -------
        function : 1d np.array
            np.array of Z values for each X,Y pair.
        """
        X, Y = point[:, 0], point[:, 1]
        rho = np.sqrt(X**2 + Y**2)
        function = np.zeros(len(point))
        nan_idx = (rho > self.Diam / 2.) + (rho < self.diam / 2.)
        function[nan_idx] = np.nan
        if self.kappa is None:
            raise ValueError("kappa must not be None for conic plot calculations")
        rho_valid = rho[~nan_idx]
        function[~nan_idx] = self.c * rho_valid**2 / (1 + (1 - self.kappa * self.c**2 * rho_valid**2)**0.5)
        return function