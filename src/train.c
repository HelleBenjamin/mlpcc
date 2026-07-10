#include "../include/config.h"
#include "../include/mlpcc.h"
#include <stdlib.h>

/* needed for training*/
float relu_derivative(float val) {
  return val > 0.0f ? 1.0f : 0.0f;
}

float sigmoid_derivative(float val) {
  return val * (1.0f - val);
}

float rand_weight(void) {
  return ((float)rand() / (float)RAND_MAX - 0.5f) * 0.2f;
}

void init_mlp(mlp *nn) {
  /* reset the nn to random state*/
  //nn->learning_rate = 0.01f;
  nn->error = 0.0f;

  for (int i = 0; i < SIZE_INPUT; i++) {
    nn->il.neuron[i] = rand_weight();
  }

  for (int layer = 0; layer < NUM_HIDDEN; layer++) {
    for (int h = 0; h < SIZE_HIDDEN; h++) {
      nn->hl.neuron[layer][h] = 0.0f;
      nn->hl.bias[layer][h] = 0.01f; /* positive bias helps relu*/
    }
  }

  for (int i = 0; i < SIZE_INPUT; i++) {
    for (int h = 0; h < SIZE_HIDDEN; h++) {
      nn->hl.input_weights[i][h] = rand_weight();
    }
  }

  for (int layer = 0; layer < NUM_HIDDEN - 1; layer++) {
    for (int j = 0; j < SIZE_HIDDEN; j++) {
      for (int h = 0; h < SIZE_HIDDEN; h++) {
        nn->hl.hidden_weights[layer][j][h] = rand_weight();
      }
    }
  }

  for (int o = 0; o < SIZE_OUTPUT; o++) {
    nn->ol.neuron[o] = 0.0f;
    nn->ol.bias[o] = 0.0f;

    for (int h = 0; h < SIZE_HIDDEN; h++) {
      nn->ol.weights[h][o] = rand_weight();
    }
  }
}

int backpropagation(mlp *nn, float target[SIZE_OUTPUT]) {
  /* single backpropagation pass*/
  /* cpu only for now*/
  float output_delta[SIZE_OUTPUT];
  float hidden_delta[NUM_HIDDEN][SIZE_HIDDEN];

  int last = NUM_HIDDEN - 1;

  nn->error = 0.0f;

  /* output delta:
    delta = output - target
  */
  for (int o = 0; o < SIZE_OUTPUT; o++) {
    float diff = nn->ol.neuron[o] - target[o];

    output_delta[o] = diff;
    nn->error += 0.5f * diff * diff;
  }

  for (int h = 0; h < SIZE_HIDDEN; h++) {
    float error = 0.0f;

    for (int o = 0; o < SIZE_OUTPUT; o++) {
      error += output_delta[o] * nn->ol.weights[h][o];
    }

    hidden_delta[last][h] = error * relu_derivative(nn->hl.neuron[last][h]);
  }

  /* only if NUM_HIDDEN > 1*/
  for (int layer = NUM_HIDDEN-2; layer >= 0; layer--) {
    for (int h = 0; h < SIZE_HIDDEN; h++) {
      float error = 0.0f;

      for (int next = 0; next < SIZE_HIDDEN; next++) {
        error += hidden_delta[layer+1][next] * nn->hl.hidden_weights[layer][h][next];
      }

      hidden_delta[layer][h] = error * relu_derivative(nn->hl.neuron[last][h]);
    }
  }

  /* update output layer*/
  for (int o = 0; o < SIZE_OUTPUT; o++) {
    nn->ol.bias[o] -= nn->learning_rate * output_delta[o];

    for (int h = 0; h < SIZE_HIDDEN; h++) {
      float grad = nn->hl.neuron[last][h] * output_delta[o];
      nn->ol.weights[h][o] -= nn->learning_rate * grad;
    }
  }

  for (int layer = 1; layer < NUM_HIDDEN; layer++) {
    for (int h = 0; h < SIZE_HIDDEN; h++) {
      nn->hl.bias[layer][h] -= nn->learning_rate * hidden_delta[layer][h];

      for (int j = 0; j < SIZE_HIDDEN; j++) {
        float grad = nn->hl.neuron[layer-1][j] * hidden_delta[layer][h];
        nn->hl.hidden_weights[layer-1][j][h] -= nn->learning_rate * grad;
      }
    }
  }

  for (int h = 0; h < SIZE_HIDDEN; h++) {
    nn->hl.bias[0][h] -= nn->learning_rate * hidden_delta[0][h];

    for (int i = 0; i < SIZE_INPUT; i++) {
      float grad = nn->il.neuron[i] * hidden_delta[0][h];
      nn->hl.input_weights[i][h] -= nn->learning_rate * grad;
    }
  }

  return 0;
}