from typing import List

from ef.field import Field
import cupy as cp

from ef.particle_array import ParticleArray


class FieldParticles(Field):
    gpu_list = (0,)
    batches = 10

    def __init__(self, name, particle_arrays):
        super().__init__(name, electric_or_magnetic='electric')
        self.particle_arrays: List[ParticleArray] = particle_arrays
        self._field_at_points = [cp.RawKernel(r'''
        extern "C" __global__
        void field_at_points(const int n_points, const int n_particles, const double charge, const double* points, const double* particles, double* forces) {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            if (tid < n_points) {
                int po = tid;
                double dx, dy, dz;
                for(int pa=0; pa<n_particles; pa++) {
                    dx = points[3*po] - particles[3*pa];
                    dy = points[3*po+1] - particles[3*pa+1];
                    dz = points[3*po+2] - particles[3*pa+2];
                    forces[3*po] += charge * dx  * pow(dx*dx + dy*dy + dz*dz, -1.5);
                    forces[3*po+1] += charge * dy * pow(dx*dx + dy*dy + dz*dz, -1.5);
                    forces[3*po+2] += charge * dz * pow(dx*dx + dy*dy + dz*dz, -1.5);
                }
            }
        }
        ''', 'field_at_points') for _ in range(len(self.gpu_list))]

    def field_at_points(self, p: ParticleArray, points, i):
        forces = cp.zeros(points.size)
        points = cp.asanyarray(points)
        n = points.shape[0]
        block = 128
        grid = (n - 1) // block + 1
        self._field_at_points[i]((grid,), (block,), (n, p.positions.shape[0], p.charge, points.ravel(order='C'), cp.asanyarray(p.positions).ravel(order='C'), forces))
        return p.charge * forces

    def get_at_points(self, positions, time):
        n_gpu = len(self.gpu_list)
        positions_slices = cp.array_split(cp.asarray(positions), n_gpu)
        index = zip(range(n_gpu), self.gpu_list, positions_slices)
        field = list(range(n_gpu))
        for i, gpu_id, pos in index:
            with cp.cuda.Device(gpu_id):
                field[i] = cp.zeros(pos.size)
                for p in self.particle_arrays:
                    field[i] = self.field_at_points(p, pos, i)
                field[i] = field[i].reshape(pos.shape)
        field = cp.concatenate([cp.asarray(f) for f in field])
        return field if hasattr(positions, 'get') else field.get()
