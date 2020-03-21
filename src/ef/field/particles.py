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
        void field_at_points(int n_points, int n_particles, const double* points, const double* particles, double* forces) {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            if (tid < n_points) {
                int po = tid;
                double dx, dy, dz;
                for(int pa=0; pa<n_particles; pa++) {
                    dx = points[3*po] - particles[3*pa];
                    dy = points[3*po+1] - particles[3*pa+1];
                    dz = points[3*po+2] - particles[3*pa+2];
                    forces[3*po] += dx * pow(dx*dx + dy*dy + dz*dz, -1.5);
                    forces[3*po+1] += dy * pow(dx*dx + dy*dy + dz*dz, -1.5);
                    forces[3*po+2] += dz * pow(dx*dx + dy*dy + dz*dz, -1.5);
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
        self._field_at_points[i]((grid,), (block,), (n, p.positions.shape[0], points.ravel(order='C'), cp.asanyarray(p.positions).ravel(order='C'), forces))
        forces = forces.reshape(points.shape)
        return p.charge * forces

    def get_at_points(self, positions, time):
        field = list(range(len(self.gpu_list)))
        for i, pair in enumerate(zip(self.gpu_list, cp.array_split(cp.asarray(positions), len(self.gpu_list)))):
            gpu_id, pos = pair
            with cp.cuda.Device(gpu_id):
                field[i] = sum(cp.nan_to_num(self.field_at_points(p, pos, i)) for p in self.particle_arrays)
        field = cp.concatenate([cp.asarray(f) for f in field])
        return field if hasattr(positions, 'get') else field.get()
