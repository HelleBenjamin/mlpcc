CC := gcc
CFLAGS := -O2 -Wall -lm

all:
	$(CC) $(CFLAGS) -o mlpcc src/main.c src/forward.c src/train.c