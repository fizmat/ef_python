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

    def field_at_points(self, p: ParticleArray, points):
        for pos in cp.array_split(cp.asarray(points), self.batches):
            diff = pos - cp.asarray(p.positions)[:, cp.newaxis, :]  # (m, n/b, 3)
            dist = cp.linalg.norm(diff, axis=-1)  # (m, n/b)
            dist[dist == 0] = 1.
            forces = diff / (dist ** 3)[..., cp.newaxis]  # (m, n/b, 3)
            yield p.charge * cp.sum(forces, axis=0)  # (n/b, 3)

    def get_at_points(self, positions, time):
        field = list(range(len(self.gpu_list)))
        for i, pair in enumerate(zip(self.gpu_list, cp.array_split(cp.asarray(positions), len(self.gpu_list)))):
            gpu_id, pos = pair
            with cp.cuda.Device(gpu_id):
                field[i] = sum(cp.nan_to_num(
                    cp.concatenate(list(self.field_at_points(p, pos)))
                ) for p in self.particle_arrays)
        field = cp.concatenate([cp.asarray(f) for f in field])
        return field if hasattr(positions, 'get') else field.get()
