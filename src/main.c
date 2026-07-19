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

/* training dataset */
char* dataset;
int num_samples = 0;
float *sample_x;
float *sample_y;

void set_input(mlp *nn, float x[SIZE_INPUT]) {
  for (int i = 0; i < SIZE_INPUT; i++) { /* copy to the nn*/
    nn->il.neuron[i] = x[i];
  }
}


void export_model(mlp *nn, char *fname) {
  FILE* f = fopen(fname, "wb");
  if (!f) {
    printf("Failed to open file");
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

void load_model(mlp *nn, char *fname) {
  FILE* f = fopen(fname, "rb");
  if (!f) {
    printf("Failed to open file");
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

void load_samples(char *fname) {
  /* read x, y samples, and num_samples*/
  /* file format
  *  int num_samples
  *  float **x;
  *  float **y;
  */
  FILE* f = fopen(fname, "rb");
  if (!f) {
    printf("Failed to open file");
    return;
  }

  fread(&num_samples, sizeof(int), 1, f); /* number of samples*/

  /* allocate and read*/
  sample_x = (float*)malloc(num_samples*SIZE_INPUT*sizeof(float));
  sample_y = (float*)malloc(num_samples*SIZE_OUTPUT*sizeof(float));

  if (!sample_x || !sample_y) {
    printf("Sample X/Y malloc failed\n");
    return;
  }

  fread(sample_x, sizeof(float), (num_samples*SIZE_INPUT), f); /* input*/
  fread(sample_y, sizeof(float), (num_samples*SIZE_OUTPUT), f); /* target*/

  printf("Dataset loaded\n");
  
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
      printf("Training model \"%s\" with parameters: epochs %d, lr %f. Dataset is \"%s\"\n", name, epochs, nn.learning_rate, dataset);
      for (int epoch = 0; epoch < epochs; epoch++) {
        float total_error = 0.0f;

        for (int s = 0; s < num_samples; s++) { /* training loop*/
          set_input(&nn, &sample_x[s*SIZE_INPUT]);

          forward_pass(&nn);
          backpropagation(&nn, &sample_y[s*SIZE_OUTPUT]);

          total_error += nn.error;
        }

        if (epoch % 10 == 0) {
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
    } else if (strncmp(argv[i], "--dataset=", 10) == 0) {
      dataset = argv[i] + 10;
      load_samples(dataset);
    } 
  }

  free(sample_x);
  free(sample_y);

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