#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../src/inline_nn.h"

TEST_CASE("Model FeedForwardSingle Test")
{
    Input input1(5);
    GRU gru1(&input1, 3);
    Model nn(&input1, &gru1);

    REQUIRE(nn.fastExecution[0].addr1 == nn.memory.data());
    REQUIRE(nn.fastExecution[1].addr1 == nn.datastream);

    gru1.memory[0] = -1;
    gru1.memory[1] = -1;
    gru1.memory[2] = 0;

    size_t hx_size = input1.size_out + gru1.memory_size;
    size_t individual_weights_size = gru1.wr_weights_size;

    REQUIRE(individual_weights_size == (5 + 3 + 1) * 3);
    REQUIRE(individual_weights_size == (hx_size + 1) * gru1.memory_size); // checking

    gru1.weights[0] = -7;
    gru1.weights[1] = 10;

    gru1.weights[individual_weights_size + hx_size] = 1;     // bias of zt for first neuron
    gru1.weights[individual_weights_size + hx_size + 1] = 1; // first memory weight (-1) * 1 of zt for first neuron

    gru1.weights[individual_weights_size * 2 + hx_size] = 1; // bias of ht for first neuron

    gru1.weights[individual_weights_size * 2 + (hx_size + 1) + hx_size] = -1; // bias of ht for second neuron

    std::vector<float> data_in = {1.0, 1.0, 1.0, 1.0, 1.0};
    float *result = nn.FeedForwardSingle(data_in.data());
    for (size_t i = 0; i < input1.size_out; i++)
    {
        REQUIRE(nn.datastream[i] == Approx(1.0f));
    }

    size_t ms = gru1.memory_size;
    size_t gru_start = input1.size_out;
    vector<size_t> breaks = {
        gru_start,
        gru_start + hx_size,              // zt
        gru_start + hx_size + ms,         // zt_inv
        gru_start + hx_size + ms * 2,     // rt
        gru_start + hx_size + ms * 3,     // candidate_start
        gru_start + hx_size * 2 + ms * 3, // candidate_start_output
        gru_start + hx_size * 2 + ms * 4, // term1
        gru_start + hx_size * 2 + ms * 5, // term2
        gru_start + hx_size * 2 + ms * 6, // output_h
    };
    for (size_t i = 0; i < nn.datastream_size; i++)
    {
        if (find(breaks.begin(), breaks.end(), i) != breaks.end())
        {
            printf("\n");
        }
        printf("nn.datastream[%zu]: %f\n", i, nn.datastream[i]);
    }

    size_t idx_memory = gru_start;

    REQUIRE(nn.datastream[idx_memory + 0] == Approx(-1.0f));
    REQUIRE(nn.datastream[idx_memory + 1] == Approx(-1.0f));
    REQUIRE(nn.datastream[idx_memory + 2] == Approx(0.0f));

    size_t idx_zt = idx_memory + hx_size;
    size_t idx_inv_zt = idx_zt + ms;

    REQUIRE(nn.datastream[idx_zt + 0] == Approx(0.047426f));
    REQUIRE(nn.datastream[idx_zt + 1] == Approx(0.5f));
    REQUIRE(nn.datastream[idx_zt + 2] == Approx(0.5f));

    REQUIRE(nn.datastream[idx_inv_zt + 0] == Approx(1.0f - nn.datastream[idx_zt + 0]));
    REQUIRE(nn.datastream[idx_inv_zt + 1] == Approx(1.0f - nn.datastream[idx_zt + 1]));
    REQUIRE(nn.datastream[idx_inv_zt + 2] == Approx(1.0f - nn.datastream[idx_zt + 2]));

    size_t idx_rt = idx_inv_zt + ms;

    REQUIRE(nn.datastream[idx_rt + 0] == Approx(0.731059f));
    REQUIRE(nn.datastream[idx_rt + 1] == Approx(0.268941f));
    REQUIRE(nn.datastream[idx_rt + 2] == Approx(0.500000f));

    size_t idx_candidate_start = idx_rt + ms;

    for (size_t i = 0; i < gru1.memory_size; i++)
    {
        REQUIRE(nn.datastream[idx_candidate_start + i] == Approx(nn.datastream[idx_rt + i] * nn.datastream[idx_memory + i]));
    }

    size_t idx_candidate_output_start = idx_candidate_start + hx_size;

    REQUIRE(nn.datastream[idx_candidate_output_start + 0] == Approx(tanhf(1.0f)));
    REQUIRE(nn.datastream[idx_candidate_output_start + 1] == Approx(tanhf(-1.0f)));
    REQUIRE(nn.datastream[idx_candidate_output_start + 2] == Approx(tanhf(0.0f)));

    size_t idx_output = idx_candidate_output_start + ms * 3;

    for (size_t i = 0; i < gru1.size_out; i++)
    {
        REQUIRE(nn.memory[i] == nn.datastream[idx_output + i]);
    }
}