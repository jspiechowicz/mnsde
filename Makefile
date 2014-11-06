CC = nvcc
OPT =--use_fast_math -O3
CURAND =-L/usr/local/cuda/lib64 -lcurand
GSLLINK =-L/usr/lib/ -lgsl -lgslcblas

all: underdamped munderdamped overdamped

underdamped: underdamped.cu
	$(CC) $(OPT) -o underdamped underdamped.cu $(CURAND) $(GSLLINK) -lm

munderdamped: munderdamped.cu
	$(CC) $(OPT) -o munderdamped munderdamped.cu $(CURAND) $(GSLLINK) -lm

overdamped: overdamped.cu
	$(CC) $(OPT) -o overdamped overdamped.cu $(CURAND) -lm

single: underdamped.cu munderdamped.cu overdamped.cu
	$(CC) $(OPT) -o underdamped underdamped.cu $(CURAND) $(GSLLINK) -lm
	$(CC) $(OPT) -o munderdamped munderdamped.cu $(CURAND) $(GSLLINK) -lm
	$(CC) $(OPT) -o overdamped overdamped.cu $(CURAND) -lm

double: double_underdamped.cu double_munderdamped.cu double_overdamped.cu
	./double.sh
	$(CC) -o underdamped double_underdamped.cu $(CURAND) $(GSLLINK) -lm
	$(CC) -o munderdamped double_munderdamped.cu $(CURAND) $(GSLLINK) -lm
	$(CC) -o overdamped double_overdamped.cu $(CURAND) -lm

