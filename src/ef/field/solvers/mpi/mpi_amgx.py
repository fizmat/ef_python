import numpy as np
import pyamgx
from mpi4py import MPI

comm = MPI.Comm.Get_parent()
size = comm.Get_size()
rank = comm.Get_rank()
print(size, rank)
conf_string = comm.recv(source=0, tag=11)
print(conf_string)
# cfg = pyamgx.Config()
# phi_vec = None
# comm.bcast(phi_vec, root=0)
# A = None
# # comm.bcast(A, root=MPI.PROC_NULL)
# cfg.create(conf_string)
# resources = pyamgx.Resources().create(cfg, comm.Clone(), 1, np.zeros(1, np.int32))
# _rhs = pyamgx.Vector().create(resources)
# _phi_vec = pyamgx.Vector().create(resources).upload(phi_vec)
# _matrix = pyamgx.Matrix().create(resources).upload_CSR(A.tocsr())