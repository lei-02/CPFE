import numpy as np

from pfem_cpfe.contracts import Case2Parameters
from pfem_cpfe.materials.case2_laws import (
    case2_slip_resistance,
    double_exponential_slip,
    kocks_mecking_increment,
    total_and_forest_density,
)


def test_double_exponential_slip_matches_oxford_branches():
    parameters = Case2Parameters.default()
    stress = np.array([0.5 * parameters.tauc0, -0.5 * parameters.tauc0, 1.1 * parameters.tauc0])

    rate = double_exponential_slip(
        stress,
        np.zeros(3),
        np.full(3, parameters.tauc0),
        293.15,
        parameters,
    )

    thermal_rate = parameters.gamma0 * np.exp(
        -(parameters.activation_energy_octahedral / parameters.boltzmann / 293.15)
        * (1.0 - 0.5**parameters.inner_exponent) ** parameters.outer_exponent
    )
    np.testing.assert_allclose(rate[:2], [thermal_rate, -thermal_rate])
    assert rate[2] == parameters.gamma0


def test_kocks_mecking_increment_matches_oxford_equation():
    parameters = Case2Parameters.default()
    forest_density = np.full(12, 0.81)
    ssd = np.full(12, 0.46)
    slip_rate = np.linspace(0.01, 0.12, 12)
    dt = 0.02

    expected = (
        parameters.k1 / parameters.burgers * np.sqrt(forest_density)
        - parameters.k2 * ssd
    ) * np.abs(slip_rate) * dt

    np.testing.assert_allclose(
        kocks_mecking_increment(ssd, forest_density, slip_rate, dt, parameters),
        expected,
    )


def test_total_and_forest_density_matches_source_matrix_operations():
    ssd = np.array([0.4, 0.6])
    gnd = np.zeros(2)
    forest = np.array([0.1, 0.2])
    forest_projection = np.eye(2)
    slip_to_screw = np.zeros((0, 2))

    total, scalar_total, forest_density = total_and_forest_density(
        ssd,
        gnd,
        forest,
        forest_projection,
        slip_to_screw,
    )

    np.testing.assert_allclose(total, [0.4, 0.6])
    assert scalar_total == 1.0
    np.testing.assert_allclose(forest_density, [0.5, 0.8])


def test_case2_slip_resistance_uses_forest_taylor_strength():
    parameters = Case2Parameters.default()
    forest_density = np.array([0.25, 1.0])

    resistance = case2_slip_resistance(forest_density, parameters)

    expected = parameters.tauc0 + parameters.c44 * parameters.burgers * parameters.geometric_factor * np.sqrt(forest_density)
    np.testing.assert_allclose(resistance, expected)
