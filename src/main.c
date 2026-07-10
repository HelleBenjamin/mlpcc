#include "../include/config.h"
#include "../include/mlpcc.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

float relu(float val) {
  return (val > 0.0f) ? val : 0.0f; /* simple as f, return if value > 0*/
}

float sigmoid(float val) {
  return 1.0f / (1.0f+expf(-val));
}

/* XOR logic gate*/
#define NUM_SAMPLES 4
float xs[NUM_SAMPLES][SIZE_INPUT] = {
  {0.0f, 0.0f},
  {0.0f, 1.0f},
  {1.0f, 0.0f},
  {1.0f, 1.0f},
};

float ys[NUM_SAMPLES][SIZE_OUTPUT] = {
  {0.0f},
  {1.0f},
  {1.0f},
  {0.0f},
};

void set_input(mlp *nn, float x[SIZE_INPUT]) {
  for (int i = 0; i < SIZE_INPUT; i++) { /* copy to the nn*/
    nn->il.neuron[i] = x[i];
  }
}


void export_model(mlp *nn, char * fname) {
  FILE* f = fopen(fname, "wb");
  if (!f) {
    printf("Faile to open file");
    return;
  }
  char *version = "mlpcc"; /* or checksum*/
  fwrite(version, 6, 1, f);

  /* write weights and biases*/
  fwrite(nn->hl.input_weights, sizeof(nn->hl.input_weights), 1, f);
  fwrite(nn->hl.hidden_weights, sizeof(nn->hl.hidden_weights), 1, f);
  fwrite(nn->hl.bias, sizeof(nn->hl.bias), 1, f);
  fwrite(nn->ol.bias, sizeof(nn->ol.bias), 1, f);
  fwrite(nn->ol.weights, sizeof(nn->ol.weights), 1, f);

  fclose(f);
}

void load_model(mlp *nn, char * fname) {
  FILE* f = fopen(fname, "rb");
  if (!f) {
    printf("Faile to open file");
    return;
  }
  char *version = "mlpcc"; /* or checksum*/
  char buf[6];
  fread(&buf, 6, 1, f);
  if (strncmp(buf, version, 6) != 0) {
    printf("Checksum mismatch\n");
  }

  /* write weights and biases*/
  fread(nn->hl.input_weights, sizeof(nn->hl.input_weights), 1, f);
  fread(nn->hl.hidden_weights, sizeof(nn->hl.hidden_weights), 1, f);
  fread(nn->hl.bias, sizeof(nn->hl.bias), 1, f);
  fread(nn->ol.bias, sizeof(nn->ol.bias), 1, f);
  fread(nn->ol.weights, sizeof(nn->ol.weights), 1, f);

  fclose(f);
}

int main(int argc, char* argv[]) {
  mlp nn;
  char *name = NULL;
  int epochs = 0;
  nn.learning_rate = 0.01f;
  for (int i = 0; i < argc; i++) {
    if (strcmp(argv[i], "--train") == 0) {
      init_mlp(&nn);
      printf("Training model \"%s\" with parameters: epochs %d, lr %f\n", name, epochs, nn.learning_rate);
      for (int epoch = 0; epoch < epochs; epoch++) {
        float total_error = 0.0f;

        for (int s = 0; s < NUM_SAMPLES; s++) { /* training loop*/
          set_input(&nn, xs[s]);

          forward_pass(&nn);
          backpropagation(&nn, ys[s]);

          total_error += nn.error;
        }

        if (epoch % 1000 == 0) {
          printf("epoch %d error=%f\n", epoch, total_error);
        }
      }
      printf("epoch %d error=%f\n", epochs, nn.error); /* latest error*/
      export_model(&nn, name);
    } else if (strncmp(argv[i], "-lr=", 4) == 0) {
      nn.learning_rate = atof(argv[i] + 4);
    } else if (strncmp(argv[i], "-e=", 3) == 0) {
      epochs = atoi(argv[i] + 3);
    } else if (strncmp(argv[i], "--name=", 7) == 0) {
      name = argv[i] + 7;
    } else if (strcmp(argv[i], "--run") == 0) {
      load_model(&nn, name);
      float inputs[SIZE_INPUT];
      for (int i = 0; i < SIZE_INPUT; i++) {
        printf("Input %d:", i);
        scanf("%f", &inputs[i]);
      }
      set_input(&nn, inputs);
      forward_pass(&nn);
      for (int o = 0; o < SIZE_OUTPUT; o++) {
        printf("%f ", nn.ol.neuron[o]);
      }
    }
  }


  /*for (int s = 0; s < NUM_SAMPLES; s++) {
    set_input(&nn, xs[s]);
    forward_pass(&nn);

    printf("x: ");
    for (int i = 0; i < SIZE_INPUT; i++) {
      printf("%f ", xs[s][i]);
    }

    printf("-> y: ");
    for (int o = 0; o < SIZE_OUTPUT; o++) {
      printf("%f ", nn.ol.neuron[o]);
    }

    printf("\n");
  }*/

  return 0;
}