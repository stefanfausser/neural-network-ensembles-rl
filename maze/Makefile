CFLAGS += -Wall -O3 -I../rlLib
LDFLAGS += -lm -lgsl -lgslcblas

all: maze

maze.o: mazePathPlanning.c
	$(CC) $(CFLAGS) -c $? -o $@

maze: maze.o ../rlLib/mlpLib.o ../rlLib/commonNeuralLib.o ../rlLib/matrixLib.o ../rlLib/tdlLib.o
	$(CC) $^ -o $@ $(CFLAGS) $(LDFLAGS)

clean:
	rm -rf *.o ../rlLib/*.o
	rm -rf maze
