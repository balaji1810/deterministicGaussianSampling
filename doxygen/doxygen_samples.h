/**
 * \page page_samples LCD Sample Gallery
 *
 * These plots show deterministic LCD / Dirac-mixture approximations of 2-D
 * Gaussian densities. In each panel the shaded field is the target Gaussian
 * density and the red points are the optimized
 * deterministic sample (Dirac) locations produced by the
 * \ref page_gm_short "gm_to_dirac_short" sampler.
 * 
 * Within the framework of the Gaussian-to-Dirac approximation, the covariance matrix \f$ \Sigma \in R^{N×N} \f$ is first spectrally decomposed:
 * 
 * <center>\f$ \Sigma = Q \Lambda Q^⊤ \f$</center>
 * 
 * where \f$ Q \f$ is orthogonal and \f$ \Lambda = \text{diag}(\lambda_1, \ldots, \lambda_N) \f$ contains the positive eigenvalues.
 * This transformation diagonalizes the covariance and eliminates inter-dimensional correlation, allowing optimization to be performed
 * in an uncorrelated, axis-aligned coordinate system. The optimization of the Dirac support points is thus carried out in the diagonalized
 * coordinate system with covariance \f$ \Lambda \f$. Upon completion of the optimization, the support points \f$ z_i \f$ determined in the
 * transformed space are mapped back to the original coordinate system via \f$ x_i = Q z_i \f$.
 *
 * \section page_samples_correlated Correlated covariances
 *
 * The grid sweeps both correlation and sample count for the unit-variance
 * covariance \f$ \Sigma = \begin{bmatrix} 1 & \rho \\ \rho & 1 \end{bmatrix} \f$.
 *
 * \image html samples_correlated.png "Rows: increasing correlation rho = 0.3, 0.6, 0.9. Columns: increasing sample count L = 10, 20, 40, 80." width=80%
 *
 * \section page_samples_standard Standard normal scaling
 *
 * \image html samples_standard_normal.png "Standard normal (Sigma = I) approximated with increasing sample count L = 10, 20, 30, 40, 50, 60." width=80%
 */
