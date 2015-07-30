CFLAGS += -Wall -O3 -I../rlLib
# optimized for Phenom X6 1090T
# CFLAGS += -Wall -O3 -march=amdfam10 -I../rlLib
LDFLAGS += -lm -lgsl -lgslcblas

all: sz-tetris

sz-tetris.o: sz-tetris.c
	$(CC) $(CFLAGS) -c $? -o $@

sz-tetris: sz-tetris.o ../rlLib/mlpLib.o ../rlLib/commonNeuralLib.o ../rlLib/matrixLib.o ../rlLib/tdlLib.o
	$(CC) $^ -o $@ $(CFLAGS) $(LDFLAGS)

clean:
	rm -rf *.o ../rlLib/*.o
	rm -rf sz-tetris
