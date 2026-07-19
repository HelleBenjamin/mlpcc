#ifndef CONFIG_H
#define CONFIG_H

/* neural network size*/
#define SIZE_INPUT 1024 /* input layer */
#define SIZE_HIDDEN 256 /* how many neurons for hidden layer*/
#define NUM_HIDDEN 3 /* how many hidden layers*/
#define SIZE_OUTPUT 1 /* output layer size */

/* activation functions */
/* 0 - relu */
/* 1 - sigmoid*/

typedef enum {
  TYPE_RELU,
  TYPE_SIGMOID
} actfunc;

#define ACTIVATION_HIDDEN TYPE_RELU

#define LEARNING_RATE 0.001f
#define EPOCHS 100


#endif