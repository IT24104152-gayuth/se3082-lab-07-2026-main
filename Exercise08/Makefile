CC = mpicc
CFLAGS = -O2

PROGRAMS = sum_bcast sum_scatter sum_gather sum_reduce sum_allreduce sum_scan

all: $(PROGRAMS)

sum_bcast: sum_bcast.c
	$(CC) $(CFLAGS) sum_bcast.c -o sum_bcast

sum_scatter: sum_scatter.c
	$(CC) $(CFLAGS) sum_scatter.c -o sum_scatter

sum_gather: sum_gather.c
	$(CC) $(CFLAGS) sum_gather.c -o sum_gather

sum_reduce: sum_reduce.c
	$(CC) $(CFLAGS) sum_reduce.c -o sum_reduce

sum_allreduce: sum_allreduce.c
	$(CC) $(CFLAGS) sum_allreduce.c -o sum_allreduce

sum_scan: sum_scan.c
	$(CC) $(CFLAGS) sum_scan.c -o sum_scan

run: all
	mpirun -np 4 --hostfile /shared/machinefile ./sum_bcast
	mpirun -np 4 --hostfile /shared/machinefile ./sum_scatter
	mpirun -np 4 --hostfile /shared/machinefile ./sum_gather
	mpirun -np 4 --hostfile /shared/machinefile ./sum_reduce
	mpirun -np 4 --hostfile /shared/machinefile ./sum_allreduce
	mpirun -np 4 --hostfile /shared/machinefile ./sum_scan

clean:
	rm -f $(PROGRAMS)