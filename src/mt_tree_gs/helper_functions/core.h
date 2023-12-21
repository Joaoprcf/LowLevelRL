#include <cstddef>

void normalize(float *input, size_t size)
{
    float sum = 0.0f;
    for (size_t i = 0; i < size; ++i)
    {
        sum += input[i] * input[i];
    }

    float dist = sqrt(sum);
    for (size_t i = 0; i < size; ++i)
    {
        input[i] /= dist;
    }
}

void generateNormalizedRandomWeights(float *weights, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        weights[i] = (float)rand() / (float)size;
    }
    normalize(weights, size);
}

void inverseWeights(float *dst, float *src, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        dst[i] = -src[i];
    }
}

void calculateForces(float *forces, float *weights, size_t weights_size, size_t dual_directions)
{
    size_t half_weights_size = weights_size * dual_directions;
    size_t full_weights_size = half_weights_size * 2;
    memset(forces, 0, sizeof(float) * half_weights_size);
    float *weights_force = new float[weights_size];

    for (size_t i = 0; i < dual_directions; i++)
    {
        memset(weights_force, 0, sizeof(float) * weights_size);
        for (size_t j = 0; j < dual_directions * 2; j++)
        {
            float strength = 0.0f;
            if (i == j || i + dual_directions == j)
                continue;
            float *wi = weights + i * weights_size;
            float *wj = weights + j * weights_size;
            for (size_t k = 0; k < weights_size; ++k)
            {
                float dist = wi[k] - wj[k];
                weights_force[k] = dist;
                strength += dist * dist;
            }
            for (size_t k = 0; k < weights_size; ++k)
            {
                forces[i * weights_size + k] += weights_force[k] / (strength > 0.0f ? strength : 1.0f);
            }
        }
    }
    delete[] weights_force;
}

void generateEvenlyDistributedWeights(float *allWeights, size_t weights_size, size_t dual_directions, size_t iterations = 1000)
{
    size_t half_weights_size = weights_size * dual_directions;
    size_t full_weights_size = half_weights_size * 2;
    for (size_t i = 0; i < dual_directions; i++)
    {
        generateNormalizedRandomWeights(allWeights + i * weights_size, weights_size);
    }
    // inverse weights
    for (size_t i = 0; i < dual_directions; i++)
    {
        inverseWeights(allWeights + half_weights_size + i * weights_size, allWeights + i * weights_size, weights_size);
    }
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f, 0.01f);
    float *forces = new float[half_weights_size];
    for (size_t _ = 0; _ < iterations; ++_)
    {
        calculateForces(forces, allWeights, weights_size, dual_directions);
        for (size_t i = 0; i < dual_directions; i++)
        {
            float *wi = allWeights + i * weights_size;
            normalize(forces + i * weights_size, weights_size);

            for (size_t k = 0; k < weights_size; ++k)
            {
                wi[k] += forces[i * weights_size + k] * 0.1f + distribution(generator);
            }
            normalize(wi, weights_size);
            inverseWeights(allWeights + half_weights_size + i * weights_size, wi, weights_size);
        }
    }
    for (size_t _ = 0; _ < iterations; ++_)
    {
        calculateForces(forces, allWeights, weights_size, dual_directions);
        for (size_t i = 0; i < dual_directions; i++)
        {
            float *wi = allWeights + i * weights_size;
            normalize(forces + i * weights_size, weights_size);

            for (size_t k = 0; k < weights_size; ++k)
            {
                wi[k] += forces[i * weights_size + k] * 0.1f;
            }
            normalize(wi, weights_size);
            inverseWeights(allWeights + half_weights_size + i * weights_size, wi, weights_size);
        }
    }
    delete[] forces;
}