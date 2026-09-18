module case2_reference_module
  implicit none
  private
  integer, parameter :: dp = kind(1.0d0)
  integer, parameter :: nslip = 12
  integer, parameter :: nscrew = 6
  real(dp), parameter :: c11 = 289.0d3
  real(dp), parameter :: c12 = 124.0d3
  real(dp), parameter :: c44 = 83.0d3
  real(dp), parameter :: rho0 = 0.46d0
  real(dp), parameter :: tauc0 = 206.0d0
  real(dp), parameter :: burgers = 2.53d-4
  real(dp), parameter :: k1 = 0.129d0
  real(dp), parameter :: k2 = 10.0d0
  real(dp), parameter :: gamma0 = 1.0d6
  real(dp), parameter :: pexp = 0.7d0
  real(dp), parameter :: qexp = 1.3d0
  real(dp), parameter :: activation = 2.97d-19
  real(dp), parameter :: boltzmann = 1.380649d-23
  real(dp), parameter :: geometric_factor = 0.3d0
  real(dp), parameter :: smallnum = 1.0d-20
  public :: case2_history

contains

  subroutine case2_history(f_history, dt_history, temperature, nstep, &
      stress_history, fp_history, orientation_history, slip_history, &
      ssd_history, crss_history, status)
    integer, intent(in) :: nstep
    real(dp), intent(in) :: f_history(3, 3, nstep)
    real(dp), intent(in) :: dt_history(nstep)
    real(dp), intent(in) :: temperature
    real(dp), intent(out) :: stress_history(6, nstep)
    real(dp), intent(out) :: fp_history(3, 3, nstep)
    real(dp), intent(out) :: orientation_history(3, 3, nstep)
    ! Use the literal public-interface extent because f2py cannot emit a C
    ! wrapper for the module-private nslip parameter.
    real(dp), intent(out) :: slip_history(12, nstep)
    real(dp), intent(out) :: ssd_history(12, nstep)
    real(dp), intent(out) :: crss_history(12, nstep)
    integer, intent(out) :: status(nstep)
    real(dp) :: f_previous(3, 3), stress(6), fp(3, 3), orientation(3, 3)
    real(dp) :: slip(nslip), ssd(nslip), crss(nslip), tangent(6, 6)
    integer :: step

    call identity3(f_previous)
    call identity3(fp)
    call identity3(orientation)
    stress = 0.0_dp
    slip = 0.0_dp
    ssd = rho0
    crss = tauc0
    status = 0

    do step = 1, nstep
      call case2_step(f_previous, f_history(:, :, step), dt_history(step), &
          temperature, stress, fp, orientation, slip, ssd, crss, tangent, &
          status(step))
      stress_history(:, step) = stress
      fp_history(:, :, step) = fp
      orientation_history(:, :, step) = orientation
      slip_history(:, step) = slip
      ssd_history(:, step) = ssd
      crss_history(:, step) = crss
      if (status(step) /= 0) return
      f_previous = f_history(:, :, step)
    end do
  end subroutine case2_history


  subroutine case2_step(f_previous, deformation, dt, temperature, stress, &
      fp, orientation, slip, ssd, crss, tangent, status)
    real(dp), intent(in) :: f_previous(3, 3), deformation(3, 3), dt, temperature
    real(dp), intent(inout) :: stress(6), fp(3, 3), orientation(3, 3)
    real(dp), intent(inout) :: slip(nslip), ssd(nslip), crss(nslip)
    real(dp), intent(out) :: tangent(6, 6)
    integer, intent(out) :: status
    real(dp) :: inverse_f(3, 3), determinant, velocity(3, 3), strain_tensor(3, 3)
    real(dp) :: strain_increment(6), stiffness(6, 6), trial_stress(6)
    real(dp) :: schmid(nslip, 3, 3), schmid_vector(nslip, 6)
    real(dp) :: forest_projection(nslip, nslip + nscrew)
    real(dp) :: slip_to_screw(nscrew, nslip), forest_old(nslip)
    real(dp) :: forest_new(nslip), resistance(nslip), resistance_new(nslip)
    real(dp) :: stress_old(6), stress_iterate(6), ssd_old(nslip), new_ssd(nslip)
    real(dp) :: slip_rate(nslip), plastic_velocity(3, 3), plastic_rate(6)
    real(dp) :: fp_old(3, 3), fp_increment(3, 3), fp_determinant
    real(dp) :: orientation_old(3, 3), elastic_velocity(3, 3), spin(3, 3)
    real(dp) :: identity(3, 3), resistance_change
    integer :: outer, inner_status

    status = 0
    tangent = 0.0_dp
    if (dt <= 0.0_dp .or. temperature <= 0.0_dp) then
      status = 1
      return
    end if
    call inverse3(deformation, inverse_f, determinant, inner_status)
    if (inner_status /= 0 .or. determinant <= smallnum) then
      status = 2
      return
    end if

    velocity = matmul((deformation - f_previous) / dt, inverse_f)
    strain_tensor = 0.5_dp * (velocity + transpose(velocity)) * dt
    call strain_to_voigt(strain_tensor, strain_increment)
    call cubic_stiffness(orientation, stiffness)
    stress_old = stress
    trial_stress = stress_old + matmul(stiffness, strain_increment)
    call build_schmid(orientation, schmid, schmid_vector)
    call density_operators(forest_projection, slip_to_screw)
    ssd_old = ssd
    call forest_density(ssd_old, forest_projection, slip_to_screw, forest_old)
    call slip_resistance(forest_old, resistance)
    stress_iterate = stress_old
    slip_rate = 0.0_dp
    plastic_velocity = 0.0_dp

    do outer = 1, 200
      call solve_stress(trial_stress, stress_iterate, stiffness, schmid, &
          schmid_vector, resistance, dt, temperature, stress_iterate, &
          slip_rate, plastic_velocity, plastic_rate, tangent, inner_status)
      if (inner_status /= 0) then
        status = 10 + inner_status
        return
      end if
      new_ssd = ssd_old + (k1 / burgers * sqrt(forest_old) - k2 * ssd_old) &
          * abs(slip_rate) * dt
      if (any(new_ssd < 0.0_dp)) then
        status = 20
        return
      end if
      call forest_density(new_ssd, forest_projection, slip_to_screw, forest_new)
      call slip_resistance(forest_new, resistance_new)
      resistance_change = maxval(abs(resistance_new - resistance))
      resistance = resistance_new
      if (resistance_change < 1.0d-4) exit
    end do
    if (outer > 200) then
      status = 21
      return
    end if

    call identity3(identity)
    fp_old = fp
    fp_increment = matmul(inverse_f, matmul(plastic_velocity, deformation)) * dt
    fp = matmul(fp_old, identity + fp_increment)
    fp_determinant = determinant3(fp)
    if (fp_determinant <= smallnum) then
      fp = fp_old
      status = 22
      return
    end if
    fp = fp / fp_determinant ** (1.0_dp / 3.0_dp)

    orientation_old = orientation
    elastic_velocity = velocity - plastic_velocity
    spin = 0.5_dp * (elastic_velocity - transpose(elastic_velocity))
    orientation = matmul(identity + spin * dt, orientation_old)
    stress = stress_iterate
    slip = slip + slip_rate * dt
    ssd = new_ssd
    crss = resistance
  end subroutine case2_step


  subroutine solve_stress(trial_stress, initial_stress, stiffness, schmid, &
      schmid_vector, resistance, dt, temperature, stress, slip_rate, &
      plastic_velocity, plastic_rate, tangent, status)
    real(dp), intent(in) :: trial_stress(6), initial_stress(6), stiffness(6, 6)
    real(dp), intent(in) :: schmid(nslip, 3, 3), schmid_vector(nslip, 6)
    real(dp), intent(in) :: resistance(nslip), dt, temperature
    real(dp), intent(out) :: stress(6), slip_rate(nslip)
    real(dp), intent(out) :: plastic_velocity(3, 3), plastic_rate(6)
    real(dp), intent(out) :: tangent(6, 6)
    integer, intent(out) :: status
    real(dp) :: resolved(nslip), derivative(nslip), plastic_stretch(3, 3)
    real(dp) :: plastic_increment(6), plastic_jacobian(6, 6)
    real(dp) :: residual_jacobian(6, 6), inverse_jacobian(6, 6)
    real(dp) :: residual(6), delta_stress(6), identity(6, 6)
    integer :: iteration, system, row, column, inverse_status

    stress = initial_stress
    resolved = matmul(schmid_vector, stress)
    call identity6(identity)
    tangent = stiffness
    status = 0
    do iteration = 1, 200
      call slip_rate_derivative(resolved, resistance, temperature, slip_rate, derivative)
      plastic_velocity = 0.0_dp
      do system = 1, nslip
        plastic_velocity = plastic_velocity + slip_rate(system) * schmid(system, :, :)
      end do
      plastic_stretch = 0.5_dp * (plastic_velocity + transpose(plastic_velocity))
      call strain_to_voigt(plastic_stretch, plastic_rate)
      plastic_increment = plastic_rate * dt
      plastic_jacobian = 0.0_dp
      do system = 1, nslip
        do row = 1, 6
          do column = 1, 6
            plastic_jacobian(row, column) = plastic_jacobian(row, column) + &
                dt * derivative(system) * schmid_vector(system, row) * &
                schmid_vector(system, column)
          end do
        end do
      end do
      residual_jacobian = identity + matmul(stiffness, plastic_jacobian)
      call inverse6(residual_jacobian, inverse_jacobian, inverse_status)
      if (inverse_status /= 0) then
        status = 1
        return
      end if
      tangent = matmul(inverse_jacobian, stiffness)
      residual = trial_stress - stress - matmul(stiffness, plastic_increment)
      delta_stress = matmul(inverse_jacobian, residual)
      stress = stress + delta_stress
      resolved = matmul(schmid_vector, stress)
      if (sqrt(sum(residual * residual)) < 1.0d-8) return
    end do
    status = 2
  end subroutine solve_stress


  subroutine slip_rate_derivative(resolved, resistance, temperature, rate, derivative)
    real(dp), intent(in) :: resolved(nslip), resistance(nslip), temperature
    real(dp), intent(out) :: rate(nslip), derivative(nslip)
    real(dp) :: ratio, barrier, sign_tau
    integer :: system

    rate = 0.0_dp
    derivative = 0.0_dp
    do system = 1, nslip
      ratio = abs(resolved(system)) / resistance(system)
      sign_tau = sign(1.0_dp, resolved(system))
      if (ratio >= 1.0_dp) then
        rate(system) = sign_tau * gamma0
      else if (ratio /= 0.0_dp) then
        barrier = (1.0_dp - ratio ** pexp) ** qexp
        rate(system) = sign_tau * gamma0 * exp(-activation / boltzmann / &
            temperature * barrier)
        if (abs(rate(system)) > smallnum) then
          derivative(system) = abs(rate(system)) * activation / boltzmann / &
              temperature * qexp * (1.0_dp - ratio ** pexp) ** &
              (qexp - 1.0_dp) * pexp / resistance(system) * &
              ratio ** (pexp - 1.0_dp)
        else
          rate(system) = 0.0_dp
        end if
      end if
    end do
  end subroutine slip_rate_derivative


  subroutine initialize_systems(directions, normals)
    real(dp), intent(out) :: directions(nslip, 3), normals(nslip, 3)
    integer :: system

    directions = reshape((/ &
        1.0_dp, -1.0_dp, 0.0_dp,  0.0_dp, 1.0_dp, -1.0_dp, &
        1.0_dp, 0.0_dp, -1.0_dp,  1.0_dp, 1.0_dp, 0.0_dp, &
        0.0_dp, 1.0_dp, -1.0_dp,  1.0_dp, 0.0_dp, 1.0_dp, &
        1.0_dp, 1.0_dp, 0.0_dp,  0.0_dp, 1.0_dp, 1.0_dp, &
        1.0_dp, 0.0_dp, -1.0_dp,  1.0_dp, -1.0_dp, 0.0_dp, &
        0.0_dp, 1.0_dp, 1.0_dp,  1.0_dp, 0.0_dp, 1.0_dp /), &
        shape(directions), order=(/2, 1/))
    normals = reshape((/ &
        1.0_dp, 1.0_dp, 1.0_dp,  1.0_dp, 1.0_dp, 1.0_dp, &
        1.0_dp, 1.0_dp, 1.0_dp, -1.0_dp, 1.0_dp, 1.0_dp, &
       -1.0_dp, 1.0_dp, 1.0_dp, -1.0_dp, 1.0_dp, 1.0_dp, &
        1.0_dp,-1.0_dp, 1.0_dp,  1.0_dp,-1.0_dp, 1.0_dp, &
        1.0_dp,-1.0_dp, 1.0_dp,  1.0_dp, 1.0_dp,-1.0_dp, &
        1.0_dp, 1.0_dp,-1.0_dp,  1.0_dp, 1.0_dp,-1.0_dp /), &
        shape(normals), order=(/2, 1/))
    do system = 1, nslip
      directions(system, :) = directions(system, :) / &
          sqrt(sum(directions(system, :) ** 2))
      normals(system, :) = normals(system, :) / sqrt(sum(normals(system, :) ** 2))
    end do
  end subroutine initialize_systems


  subroutine build_schmid(orientation, schmid, vectors)
    real(dp), intent(in) :: orientation(3, 3)
    real(dp), intent(out) :: schmid(nslip, 3, 3), vectors(nslip, 6)
    real(dp) :: directions(nslip, 3), normals(nslip, 3)
    real(dp) :: direction(3), normal(3)
    integer :: system, i, j

    call initialize_systems(directions, normals)
    do system = 1, nslip
      direction = matmul(orientation, directions(system, :))
      normal = matmul(orientation, normals(system, :))
      do i = 1, 3
        do j = 1, 3
          schmid(system, i, j) = direction(i) * normal(j)
        end do
      end do
      vectors(system, 1) = schmid(system, 1, 1)
      vectors(system, 2) = schmid(system, 2, 2)
      vectors(system, 3) = schmid(system, 3, 3)
      vectors(system, 4) = schmid(system, 1, 2) + schmid(system, 2, 1)
      vectors(system, 5) = schmid(system, 1, 3) + schmid(system, 3, 1)
      vectors(system, 6) = schmid(system, 2, 3) + schmid(system, 3, 2)
    end do
  end subroutine build_schmid


  subroutine density_operators(forest_projection, slip_to_screw)
    real(dp), intent(out) :: forest_projection(nslip, nslip + nscrew)
    real(dp), intent(out) :: slip_to_screw(nscrew, nslip)
    real(dp) :: directions(nslip, 3), normals(nslip, 3)
    real(dp) :: lines(nslip + nscrew, 3), result
    integer, parameter :: screw_systems(nscrew) = (/1, 2, 3, 4, 6, 8/)
    integer :: i, j

    call initialize_systems(directions, normals)
    do i = 1, nslip
      call cross_product(directions(i, :), normals(i, :), lines(i, :))
    end do
    do i = 1, nscrew
      lines(nslip + i, :) = directions(screw_systems(i), :)
    end do
    do i = 1, nslip
      do j = 1, nslip + nscrew
        forest_projection(i, j) = abs(dot_product(normals(i, :), lines(j, :)))
      end do
    end do
    slip_to_screw = 0.0_dp
    do i = 1, nscrew
      do j = 1, nslip
        result = dot_product(directions(screw_systems(i), :), directions(j, :))
        if (abs(result) > 0.99_dp) slip_to_screw(i, j) = 0.5_dp
      end do
    end do
  end subroutine density_operators


  subroutine forest_density(ssd, forest_projection, slip_to_screw, forest)
    real(dp), intent(in) :: ssd(nslip)
    real(dp), intent(in) :: forest_projection(nslip, nslip + nscrew)
    real(dp), intent(in) :: slip_to_screw(nscrew, nslip)
    real(dp), intent(out) :: forest(nslip)

    forest = matmul(forest_projection(:, 1:nslip), ssd) + &
        matmul(forest_projection(:, nslip + 1:nslip + nscrew), &
        matmul(slip_to_screw, ssd))
  end subroutine forest_density


  subroutine slip_resistance(forest, resistance)
    real(dp), intent(in) :: forest(nslip)
    real(dp), intent(out) :: resistance(nslip)

    resistance = tauc0 + c44 * burgers * geometric_factor * sqrt(forest)
  end subroutine slip_resistance


  subroutine cubic_stiffness(rotation, stiffness)
    real(dp), intent(in) :: rotation(3, 3)
    real(dp), intent(out) :: stiffness(6, 6)
    real(dp) :: crystal(3, 3, 3, 3), sample(3, 3, 3, 3), value
    integer, parameter :: first(6) = (/1, 2, 3, 1, 1, 2/)
    integer, parameter :: second(6) = (/1, 2, 3, 2, 3, 3/)
    integer :: i, j, k, l, ii, jj, kk, ll, row, column

    crystal = 0.0_dp
    do i = 1, 3
      crystal(i, i, i, i) = c11
      do j = 1, 3
        if (i /= j) crystal(i, i, j, j) = c12
      end do
    end do
    do i = 1, 3
      do j = i + 1, 3
        crystal(i, j, i, j) = c44
        crystal(i, j, j, i) = c44
        crystal(j, i, i, j) = c44
        crystal(j, i, j, i) = c44
      end do
    end do
    sample = 0.0_dp
    do i = 1, 3
      do j = 1, 3
        do k = 1, 3
          do l = 1, 3
            value = 0.0_dp
            do ii = 1, 3
              do jj = 1, 3
                do kk = 1, 3
                  do ll = 1, 3
                    value = value + rotation(i, ii) * rotation(j, jj) * &
                        rotation(k, kk) * rotation(l, ll) * crystal(ii, jj, kk, ll)
                  end do
                end do
              end do
            end do
            sample(i, j, k, l) = value
          end do
        end do
      end do
    end do
    do row = 1, 6
      do column = 1, 6
        stiffness(row, column) = sample(first(row), second(row), &
            first(column), second(column))
      end do
    end do
  end subroutine cubic_stiffness


  subroutine strain_to_voigt(tensor, vector)
    real(dp), intent(in) :: tensor(3, 3)
    real(dp), intent(out) :: vector(6)

    vector(1) = tensor(1, 1)
    vector(2) = tensor(2, 2)
    vector(3) = tensor(3, 3)
    vector(4) = 2.0_dp * tensor(1, 2)
    vector(5) = 2.0_dp * tensor(1, 3)
    vector(6) = 2.0_dp * tensor(2, 3)
  end subroutine strain_to_voigt


  subroutine cross_product(left, right, result)
    real(dp), intent(in) :: left(3), right(3)
    real(dp), intent(out) :: result(3)

    result(1) = left(2) * right(3) - left(3) * right(2)
    result(2) = left(3) * right(1) - left(1) * right(3)
    result(3) = left(1) * right(2) - left(2) * right(1)
  end subroutine cross_product


  pure real(dp) function determinant3(matrix)
    real(dp), intent(in) :: matrix(3, 3)

    determinant3 = matrix(1, 1) * (matrix(2, 2) * matrix(3, 3) - &
        matrix(2, 3) * matrix(3, 2)) - matrix(1, 2) * &
        (matrix(2, 1) * matrix(3, 3) - matrix(2, 3) * matrix(3, 1)) + &
        matrix(1, 3) * (matrix(2, 1) * matrix(3, 2) - &
        matrix(2, 2) * matrix(3, 1))
  end function determinant3


  subroutine inverse3(matrix, inverse, determinant, status)
    real(dp), intent(in) :: matrix(3, 3)
    real(dp), intent(out) :: inverse(3, 3), determinant
    integer, intent(out) :: status

    determinant = determinant3(matrix)
    if (abs(determinant) <= smallnum) then
      inverse = 0.0_dp
      status = 1
      return
    end if
    inverse(1, 1) = matrix(2, 2) * matrix(3, 3) - matrix(2, 3) * matrix(3, 2)
    inverse(1, 2) = matrix(1, 3) * matrix(3, 2) - matrix(1, 2) * matrix(3, 3)
    inverse(1, 3) = matrix(1, 2) * matrix(2, 3) - matrix(1, 3) * matrix(2, 2)
    inverse(2, 1) = matrix(2, 3) * matrix(3, 1) - matrix(2, 1) * matrix(3, 3)
    inverse(2, 2) = matrix(1, 1) * matrix(3, 3) - matrix(1, 3) * matrix(3, 1)
    inverse(2, 3) = matrix(1, 3) * matrix(2, 1) - matrix(1, 1) * matrix(2, 3)
    inverse(3, 1) = matrix(2, 1) * matrix(3, 2) - matrix(2, 2) * matrix(3, 1)
    inverse(3, 2) = matrix(1, 2) * matrix(3, 1) - matrix(1, 1) * matrix(3, 2)
    inverse(3, 3) = matrix(1, 1) * matrix(2, 2) - matrix(1, 2) * matrix(2, 1)
    inverse = inverse / determinant
    status = 0
  end subroutine inverse3


  subroutine inverse6(matrix, inverse, status)
    real(dp), intent(in) :: matrix(6, 6)
    real(dp), intent(out) :: inverse(6, 6)
    integer, intent(out) :: status
    real(dp) :: augmented(6, 12), pivot_value, factor, temporary(12)
    integer :: row, column, pivot

    augmented(:, 1:6) = matrix
    augmented(:, 7:12) = 0.0_dp
    do row = 1, 6
      augmented(row, 6 + row) = 1.0_dp
    end do
    do column = 1, 6
      pivot = column - 1 + maxloc(abs(augmented(column:6, column)), dim=1)
      if (abs(augmented(pivot, column)) <= smallnum) then
        inverse = 0.0_dp
        status = 1
        return
      end if
      if (pivot /= column) then
        temporary = augmented(column, :)
        augmented(column, :) = augmented(pivot, :)
        augmented(pivot, :) = temporary
      end if
      pivot_value = augmented(column, column)
      augmented(column, :) = augmented(column, :) / pivot_value
      do row = 1, 6
        if (row /= column) then
          factor = augmented(row, column)
          augmented(row, :) = augmented(row, :) - factor * augmented(column, :)
        end if
      end do
    end do
    inverse = augmented(:, 7:12)
    status = 0
  end subroutine inverse6


  subroutine identity3(identity)
    real(dp), intent(out) :: identity(3, 3)
    integer :: i

    identity = 0.0_dp
    do i = 1, 3
      identity(i, i) = 1.0_dp
    end do
  end subroutine identity3


  subroutine identity6(identity)
    real(dp), intent(out) :: identity(6, 6)
    integer :: i

    identity = 0.0_dp
    do i = 1, 6
      identity(i, i) = 1.0_dp
    end do
  end subroutine identity6

end module case2_reference_module
