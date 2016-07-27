CFLAGS += -Wall -Wextra -Werror -O2

nnet1: nnet1.o

nnet2: nnet2.o

clean:
	rm -rf *.o
