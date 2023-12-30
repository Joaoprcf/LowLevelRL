

void divideWeightsAndBias(float *dst, const float *origin, size_t size_in, size_t size_out)
{
    size_t total_elements = (size_in + 1) * size_out;
    size_t weights_count = size_in * size_out;

    size_t idx = 0;
    for (size_t i = 0; i < size_out; i++)
    {
        for (size_t j = 0; j < size_in; j++, idx++)
        {
            dst[j * size_out + i] = origin[idx + i];
        }
        dst[weights_count + i] = origin[idx + i];
    }
}

void mergeWeightsAndBias(float *dst, const float *origin, size_t size_in, size_t size_out)
{
    size_t total_elements = (size_in + 1) * size_out;
    size_t weights_count = size_in * size_out;

    size_t idx = 0;
    for (size_t i = 0; i < size_out; i++)
    {
        for (size_t j = 0; j < size_in; j++, idx++)
        {
            dst[idx + i] = origin[j * size_out + i];
        }
    }

    for (size_t biax_idx = size_in; idx < total_elements; idx++, biax_idx += size_in + 1)
    {
        dst[biax_idx] = origin[idx];
    }
}