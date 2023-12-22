#include <cstddef>
#include <random>

void normalize(float *input, size_t size)
{
    float sum = 0.0f;
    for (size_t i = 0; i < size; ++i)
    {
        sum += input[i] * input[i];
    }

    float dist = sqrtf(sum);
    for (size_t i = 0; i < size; ++i)
    {
        input[i] /= dist;
    }
}

void generateNormalizedRandomWeights(float *weights, size_t size)
{

    std::random_device rd;                           // Random number engine (seed)
    std::mt19937 gen(rd());                          // Mersenne Twister generator
    std::uniform_real_distribution<> dis(-1.0, 1.0); // Distribution between -1 and 1

    for (size_t i = 0; i < size; ++i)
    {
        weights[i] = dis(gen);
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

float calculateAngle(float *coord1, float *coord2, size_t dimensions)
{

    // Calculate dot product
    float dotProduct = 0.0f;
    for (size_t i = 0; i < dimensions; ++i)
    {
        dotProduct += coord1[i] * coord2[i];
    }

    // Calculate magnitudes of the two vectors
    float magnitude1 = 0.0f, magnitude2 = 0.0f;
    for (size_t i = 0; i < dimensions; ++i)
    {
        magnitude1 += coord1[i] * coord1[i];
        magnitude2 += coord2[i] * coord2[i];
    }
    magnitude1 = sqrt(magnitude1);
    magnitude2 = sqrt(magnitude2);

    // Prevent division by zero
    if (magnitude1 == 0.0f || magnitude2 == 0.0f)
    {
        throw std::invalid_argument("Vector magnitude cannot be zero");
    }

    // Calculate the cosine of the angle
    float cosAngle = dotProduct / (magnitude1 * magnitude2);

    // Clamp cosAngle to the range [-1, 1] to avoid NaN errors due to floating point inaccuracies
    cosAngle = fmax(fmin(cosAngle, 1.0f), -1.0f);

    // Calculate the angle in radians and then convert to degrees
    float angle = acos(cosAngle); // Angle in radians

    return angle;
}

float calculateDistributionAngleError(float *weights, size_t size, float dual_directions)
{
    float max_angle = 0.0f;
    float min_angle = M_PI / 2.0f;
    for (size_t i = 0; i < dual_directions * 2; ++i)
    {
        float min_local_angle = M_PI / 2.0f;
        for (size_t j = 0; j < dual_directions * 2; ++j)
        {
            if (i == j)
                continue;
            float angle = calculateAngle(weights + i * size, weights + j * size, size);
            min_local_angle = std::min(min_local_angle, angle);
        }
        // printf("min_local_angle: %f\n", min_local_angle * 180 / M_PI);
        max_angle = std::max(max_angle, min_local_angle);
        min_angle = std::min(min_angle, min_local_angle);
    }
    printf("max_angle: %f\n", max_angle * 180 / M_PI);
    printf("min_angle: %f\n", min_angle * 180 / M_PI);
    return max_angle - min_angle;
}

void calculateForces(float *forces, float *weights, size_t weights_size, size_t dual_directions)
{
    size_t half_weights_size = weights_size * dual_directions;
    // size_t full_weights_size = half_weights_size * 2;
    memset(forces, 0, sizeof(float) * half_weights_size);
    float *weights_force = new float[weights_size];

    for (size_t i = 0; i < dual_directions; i++)
    {
        memset(weights_force, 0, sizeof(float) * weights_size);
        float *wi = weights + i * weights_size;
        float *fi = forces + i * weights_size;
        for (size_t j = 0; j < dual_directions * 2; j++)
        {
            if (j <= i || i + dual_directions == j)
                continue;
            float strength = 0.0f;
            float *wj = weights + j * weights_size;
            float *fj = forces + j * weights_size;
            for (size_t k = 0; k < weights_size; ++k)
            {
                float dist = wi[k] - wj[k];
                weights_force[k] = dist;
                strength += dist * dist;
            }
            strength = 1.0f / (strength > 0.0f ? strength : 1.0f);
            for (size_t k = 0; k < weights_size; ++k)
            {
                float final_force = weights_force[k] * strength;
                fi[k] += final_force;
                if (j < dual_directions)
                {
                    fj[k] -= final_force;
                }
            }
        }
    }
    delete[] weights_force;
}

void applyForces(float *allWeights, float *forces, size_t weights_size, size_t dual_directions, std::default_random_engine &generator, bool use_noise = false)
{
    std::normal_distribution<float> distribution(0.0f, 0.01f);
    size_t half_weights_size = weights_size * dual_directions;
    for (size_t i = 0; i < dual_directions; i++)
    {
        float *wi = allWeights + i * weights_size;
        normalize(forces + i * weights_size, weights_size);
        if (use_noise)
        {
            for (size_t k = 0; k < weights_size; ++k)
            {
                wi[k] += forces[i * weights_size + k] * 0.1f + distribution(generator);
            }
        }
        else
        {
            for (size_t k = 0; k < weights_size; ++k)
            {
                wi[k] += forces[i * weights_size + k] * 0.1f;
            }
        }
        normalize(wi, weights_size);
        inverseWeights(allWeights + half_weights_size + i * weights_size, wi, weights_size);
    }
}

void generateEvenlyDistributedWeights(float *allWeights, size_t weights_size, size_t dual_directions, size_t iterations = 1000)
{
    size_t half_weights_size = weights_size * dual_directions;
    // size_t full_weights_size = half_weights_size * 2;
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
    float *forces = new float[half_weights_size];
    for (size_t it = 0; it < iterations * 2; ++it)
    {
        calculateForces(forces, allWeights, weights_size, dual_directions);
        applyForces(allWeights, forces, weights_size, dual_directions, generator, it < iterations);
    }
    delete[] forces;
}