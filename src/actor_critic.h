#pragma once
#include "helper_functions/core.h"
#include "pipeline_builder/core.h"
#include "game_examples.h"
#include "grs_optimizers/core.h"
#include "environment/core.h"
#include <vector>
#include <random>
#include <curand.h>
#include <curand_kernel.h>
using namespace std;

struct ActorCritic : stairs(stairs)
{
    ActorCritic(Model * actor, Model * critic)
    {
        actor_builder = new PipelineBuilder(nn);
        critic_builder = new PipelineBuilder(nn);
    }
}