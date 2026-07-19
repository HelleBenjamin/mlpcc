#include "../include/config.h"
#include "../include/mlpcc.h"


/* forward propagation */
int forward_pass(mlp *nn) {
  /* input to hidden layer 0*/
  for (int i = 0; i < SIZE_HIDDEN; i++) {
    float sum = nn->hl.bias[0][i]; /* sum of each neuron*/
    
    /* calculate each input connection*/
    for (int j = 0; j < SIZE_INPUT; j++) {
      sum += nn->il.neuron[j] * nn->hl.input_weights[j][i];
    }

    nn->hl.neuron[0][i] = relu(sum);
  }

  /* hidden layer n-1 to hidden layer n*/
  for (int layer = 1; layer < NUM_HIDDEN; layer++) {
    for (int i = 0; i < SIZE_HIDDEN; i++) {
      float sum = nn->hl.bias[layer][i];

      for (int j = 0; j < SIZE_HIDDEN; j++) {
        sum += nn->hl.neuron[layer-1][j] * nn->hl.hidden_weights[layer-1][j][i];
      }

      nn->hl.neuron[layer][i] = relu(sum);
    }
  }

  /* last hidden layer to output*/
  for (int i = 0; i < SIZE_OUTPUT; i++) {
    float sum = nn->ol.bias[i];
    
    for (int j = 0; j < SIZE_HIDDEN; j++) {
      sum += nn->hl.neuron[NUM_HIDDEN-1][j] * nn->ol.weights[j][i];
    }

    nn->ol.neuron[i] = sum;
  }

  return 0;
}