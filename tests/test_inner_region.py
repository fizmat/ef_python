import numpy as np
import pytest

from ef.config.components import Box
from ef.inner_region import InnerRegion
from ef.particle_array import ParticleArray
from ef.util.testing import assert_dataclass_eq


@pytest.mark.usefixtures('backend')
class TestInnerRegion:
    def test_init(self):
        ir = InnerRegion('test', Box())
        assert ir.name == 'test'
        assert_dataclass_eq(ir.shape, Box())
        assert ir.total_absorbed_particles == 0
        assert ir.total_absorbed_charge == 0
        assert ir.inverted == False

    def test_absorb_charge(self):
        particles = ParticleArray([1], -2.0, 1.0, (0, 0, 0), np.zeros(3))
        ir = InnerRegion('test', Box())
        assert ir.total_absorbed_particles == 0
        assert ir.total_absorbed_charge == 0
        ir.collide_with_particles(particles)
        assert ir.total_absorbed_particles == 1
        assert ir.total_absorbed_charge == -2
        assert_dataclass_eq(particles, ParticleArray([], -2.0, 1.0, np.zeros((0, 3)), np.zeros((0, 3))))
        particles = ParticleArray([1], -2.0, 1.0, (10, 10, 10), np.zeros(3))
        ir = InnerRegion('test', Box())
        assert ir.total_absorbed_particles == 0
        assert ir.total_absorbed_charge == 0
        ir.collide_with_particles(particles)
        assert ir.total_absorbed_particles == 0
        assert ir.total_absorbed_charge == 0
        assert_dataclass_eq(particles, ParticleArray([1], -2.0, 1.0, [(10, 10, 10)], np.zeros((1, 3))))
        particles = ParticleArray([1, 2], -2.0, 1.0, [(0, 0, 0), (10, 10, 10)], np.zeros((2, 3)))
        ir = InnerRegion('test', Box())
        assert ir.total_absorbed_particles == 0
        assert ir.total_absorbed_charge == 0
        ir.collide_with_particles(particles)
        assert ir.total_absorbed_particles == 1
        assert ir.total_absorbed_charge == -2
        assert_dataclass_eq(particles, ParticleArray([2], -2.0, 1.0, [(10, 10, 10)], np.zeros((1, 3))))

    def test_absorb_charge_inverted(self):
        particles = ParticleArray([1], -2.0, 1.0, (0, 0, 0), np.zeros(3))
        ir = InnerRegion('test', Box(), inverted=True)
        assert ir.total_absorbed_particles == 0
        assert ir.total_absorbed_charge == 0
        ir.collide_with_particles(particles)
        assert ir.total_absorbed_particles == 0
        assert ir.total_absorbed_charge == 0
        assert_dataclass_eq(particles, ParticleArray([1], -2.0, 1.0, np.zeros((1, 3)), np.zeros((1, 3))))
        particles = ParticleArray([1], -2.0, 1.0, (10, 10, 10), np.zeros(3))
        ir = InnerRegion('test', Box(), inverted=True)
        assert ir.total_absorbed_particles == 0
        assert ir.total_absorbed_charge == 0
        ir.collide_with_particles(particles)
        assert ir.total_absorbed_particles == 1
        assert ir.total_absorbed_charge == -2
        assert_dataclass_eq(particles, ParticleArray([], -2.0, 1.0, np.zeros((0, 3)), np.zeros((0, 3))))
        particles = ParticleArray([1, 2], -2.0, 1.0, [(0, 0, 0), (10, 10, 10)], np.zeros((2, 3)))
        ir = InnerRegion('test', Box(), inverted=True)
        assert ir.total_absorbed_particles == 0
        assert ir.total_absorbed_charge == 0
        ir.collide_with_particles(particles)
        assert ir.total_absorbed_particles == 1
        assert ir.total_absorbed_charge == -2
        assert_dataclass_eq(particles, ParticleArray([1], -2.0, 1.0, [(0, 0, 0)], np.zeros((1, 3))))
