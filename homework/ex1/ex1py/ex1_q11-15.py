from threed_gaussian import *

SCALING_MAT = np.array([[0.1, 0, 0], [0, 0.5, 0], [0, 0, 2]])


def q11():
    # 11 - cov is I matrix
    mean = [0, 0, 0]
    cov = np.eye(3)
    x_y_z = np.random.multivariate_normal(mean, cov, 50000).T
    plot_3d(x_y_z, "11")
    return x_y_z


def q12(x_y_z):
    # 12 - x_y_z samples data multiplied by scaling matrix
    x_y_z_scaled = np.matmul(SCALING_MAT, x_y_z)
    plot_3d(x_y_z_scaled, "12")
    cov_scaled_samples_numeric = np.cov(x_y_z_scaled)
    cov_samples = np.cov(x_y_z)
    cov_scaled_samples_analytic = SCALING_MAT@cov_samples@SCALING_MAT.T
    print("cov of 12\n", cov_scaled_samples_numeric, "\n")
    return x_y_z_scaled


def q13(x_y_z_scaled):
    rand_ort_mat = get_orthogonal_matrix(3)
    print("random orthogonal matrix\n", rand_ort_mat)
    x_y_z_scaled_mult_orthogonal = rand_ort_mat@x_y_z_scaled
    plot_3d(x_y_z_scaled_mult_orthogonal, "13")
    print("cov of 13\n", np.cov(x_y_z_scaled_mult_orthogonal), "\n")
    return x_y_z_scaled_mult_orthogonal


def q14(x_y_z_scaled_mult_orthogonal):
    plot_2d([x_y_z_scaled_mult_orthogonal[0, :], x_y_z_scaled_mult_orthogonal[1, :]], '14')


def q15(x_y_z_scaled_mult_orthogonal):
    z = x_y_z_scaled_mult_orthogonal[2,:]
    conditional_z = (0.1 > z) & (z > -0.4)
    plot_2d([x_y_z_scaled_mult_orthogonal[0, conditional_z], x_y_z_scaled_mult_orthogonal[1, conditional_z]], '15')


def main():
    ret = q11()
    print("\n##### Question 12 #####\n")
    ret = q12(ret)
    print("\n##### Question 13 #####\n")
    ret = q13(ret)
    q14(ret)
    q15(ret)


if __name__ == "__main__":
    main()

