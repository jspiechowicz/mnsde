CC = nvcc
OPT =--use_fast_math -O3
CURAND =-L/usr/local/cuda/lib64 -lcurand

all: underdamped overdamped

underdamped: underdamped.cu
	$(CC) $(OPT) -o underdamped underdamped.cu $(CURAND) -lm

overdamped: overdamped.cu
	$(CC) $(OPT) -o overdamped overdamped.cu $(CURAND) -lm

single: underdamped.cu overdamped.cu
	$(CC) $(OPT) -o underdamped underdamped.cu $(CURAND) -lm
	$(CC) $(OPT) -o overdamped overdamped.cu $(CURAND) -lm

double: double_underdamped.cu double_overdamped.cu
	./double.sh
	$(CC) -o underdamped double_underdamped.cu $(CURAND) -lm
	$(CC) -o overdamped double_overdamped.cu $(CURAND) -lm

