import numpy

from ef.field.solvers import FieldSolver

class FieldSolverPyamgx(FieldSolver):
    def __del__(self):
        import pyamgx
        self._solver.destroy()
        self._rhs.destroy()
        self._phi_vec.destroy()
        self._matrix.destroy()
        self.resources.destroy()
        self.cfg.destroy()
        pyamgx.finalize()
        self.comm.Disconnect()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import sys
        import pyamgx
        from mpi4py import MPI
        pyamgx.initialize()
        self.cfg = pyamgx.Config()
        conf_string = f"""{{
            "config_version": 2,
            "solver": {{
                "solver": "BICGSTAB",
                "max_iters": {self.max_iter},
                "monitor_residual": 1,
                "tolerance": {self.tolerance},
                "norm": "L2",
                "print_solve_stats": 0,
                "obtain_timings": 0,
                "print_grid_stats": 0
            }},
            "communicator": "MPI"
        }}"""
        # self.comm = MPI.COMM_SELF.Spawn(sys.executable, 'field/solvers/mpi/mpi_amgx.py', maxprocs=1)
        self.comm = MPI.COMM_SELF.Clone()
        # self.comm.send(conf_string, dest=0, tag=11)
        # self.comm.Disconnect()
        self.cfg.create(conf_string)
        self.resources = pyamgx.Resources().create(self.cfg, self.comm, 1, numpy.zeros(1, numpy.int32))
        self._matrix = pyamgx.Matrix().create(self.resources)\
            .comm_from_maps(1, 1, 0, numpy.zeros(1, dtype=numpy.int32),
                            numpy.array([0], dtype=numpy.int32),
                            numpy.zeros(1, dtype=numpy.int32),
                            numpy.array([0], dtype=numpy.int32),
                            numpy.zeros(1, dtype=numpy.int32))\
            .upload_CSR(self.A.tocsr())
        self._rhs = pyamgx.Vector().create(self.resources).bind(self._matrix)
        self._phi_vec = pyamgx.Vector().create(self.resources).bind(self._matrix).upload(self.phi_vec)
        self._solver = pyamgx.Solver().create(self.resources, self.cfg)
        self._solver.setup(self._matrix)

    def eval_potential(self, charge_density, potential):
        self.init_rhs_vector(charge_density, potential)
        if hasattr(self.rhs.data, 'ptr'):
            self._rhs.upload_raw(self.rhs.data.ptr, len(self.rhs))
        else:
            self._rhs.upload(self.rhs)
        self._solver.solve(self._rhs, self._phi_vec)
        self.transfer_solution_to_spat_mesh(potential)

    def transfer_solution_to_spat_mesh(self, potential):
        if potential.xp is numpy:
            self._phi_vec.download(self.phi_vec)
            super().transfer_solution_to_spat_mesh(potential)
        else:
            buf = potential.xp.empty(int((self.mesh.n_nodes - 2).prod()))
            self._phi_vec.download_raw(buf.data.ptr)
            potential._data[1:-1, 1:-1, 1:-1] = buf.reshape(self.mesh.n_nodes - 2, order='F')

    def init_rhs_vector_in_full_domain(self, charge_density, potential):
        charge = charge_density._data
        pot = potential._data
        rhs = -4 * numpy.pi * self.mesh.cell.prod() ** 2 * charge[1:-1, 1:-1, 1:-1]
        dx, dy, dz = self.mesh.cell
        rhs[0] -= dy * dy * dz * dz * pot[0, 1:-1, 1:-1]
        rhs[-1] -= dy * dy * dz * dz * pot[-1, 1:-1, 1:-1]
        rhs[:, 0] -= dx * dx * dz * dz * pot[1:-1, 0, 1:-1]
        rhs[:, -1] -= dx * dx * dz * dz * pot[1:-1, -1, 1:-1]
        rhs[:, :, 0] -= dx * dx * dy * dy * pot[1:-1, 1:-1, 0]
        rhs[:, :, -1] -= dx * dx * dy * dy * pot[1:-1, 1:-1, -1]
        self.rhs = rhs.ravel('F')
