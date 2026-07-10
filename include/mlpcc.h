#ifndef MLPCC_H
#define MLPCC_H

#include "config.h"

typedef struct {
  float neuron[SIZE_INPUT]; /* input values*/
} input_layer;

typedef struct {
  float neuron[NUM_HIDDEN][SIZE_HIDDEN];
  float bias[NUM_HIDDEN][SIZE_HIDDEN];
  float input_weights[SIZE_INPUT][SIZE_HIDDEN];
  float hidden_weights[NUM_HIDDEN - 1][SIZE_HIDDEN][SIZE_HIDDEN];
} hidden_layer;

typedef struct {
  float neuron[SIZE_OUTPUT];
  float bias[SIZE_OUTPUT];
  float weights[SIZE_HIDDEN][SIZE_OUTPUT];
} output_layer;

typedef struct {

  float learning_rate; /* for training */
  float error;

  /* actual nn stuff*/
  input_layer il;
  hidden_layer hl;
  output_layer ol;
} mlp;


int forward_pass(mlp *nn);
int backpropagation(mlp *nn, float target[SIZE_OUTPUT]);
void init_mlp(mlp *nn);

float relu(float val);
float sigmoid(float val);

#endif