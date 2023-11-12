#include "src/inline_nn.h"

int main()
{

    Input input1(5);
    Dense dense1(&input1, 2);
    Dense dense2(&dense1, 2);
    Concatenate ct({&dense1, &dense2});
    Dense dense3(&ct, 2);
    NeuralNetwork nn(&input1, &dense3);

    dense1.weights[6] = 1;
    dense2.weights[4] = 1;

    dense3.weights[1] = 10;
    dense3.weights[3] = -7;
    dense3.weights[6] = -10;
    dense3.weights[8] = 7;

    for (size_t i = 0; i < nn.weights.size(); i++)
    {
        /* code */
        printf("nn.weights[%zu] = %.2f\n", i, nn.weights[i]);
    }

    vector<float> data_in = {1.0, 1.0, 1.0, 1.0, 1.0};
    float *result = nn.FeedForwardSingle(data_in.data());
    assert(3.0f == result[0]);
    assert(-3.0f == result[1]);

    // CUDA code goes here
    return 0;
}
