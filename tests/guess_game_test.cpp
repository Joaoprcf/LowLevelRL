#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../src/game_examples.h"

TEST_CASE("Generate Values Range")
{
    GuessGame game;
    game.generateValues();

    for (int i = 0; i < 5; ++i)
    {
        REQUIRE(game.values[i] >= -1);
        REQUIRE(game.values[i] <= 1);
    }
}

TEST_CASE("Game Step Functionality")
{
    GuessGame game(12345);
    float observation[5] = {};
    float action[2] = {0.5f, -0.5f};
    game.reset(observation);

    game.step(action, observation);

    REQUIRE(game.missing_steps == 19);

    for (int i = 0; i < 5; ++i)
    {
        REQUIRE(observation[i] == game.values[i]);
        REQUIRE(observation[i] >= -1);
        REQUIRE(observation[i] <= 1);
    }

    for (int i = 0; i < 19; i++)
    {
        game.step(action, observation);
    }

    REQUIRE(game.missing_steps == 0);
}

TEST_CASE("Game Reset Functionality")
{
    GuessGame game(12345);
    float observation[5] = {};
    float action[2] = {0.5f, -0.5f};
    game.reset(observation);
    game.step(action, observation);
    game.reset(observation);

    REQUIRE(game.reward == 0);
    REQUIRE(game.missing_steps == 20);
}

TEST_CASE("GameHard Reset Functionality")
{
    GuessGameHard game(12345);
    float observation[5] = {};
    float action[2] = {0.5f, -0.5f};
    game.reset(observation);
    game.step(action, observation);
    game.reset(observation);

    REQUIRE(game.reward == 0);
    REQUIRE(game.missing_steps == 20);
}

TEST_CASE("GameHard Step Functionality")
{
    GuessGameHard game(12345);
    float observation[5] = {};
    float action[2] = {0.5f, -0.5f};
    game.reset(observation);

    game.step(action, observation);

    REQUIRE(game.missing_steps == 19);

    for (int i = 0; i < 5; ++i)
    {
        REQUIRE(observation[i] == game.values[i]);
        REQUIRE(observation[i] >= -1);
        REQUIRE(observation[i] <= 1);
    }

    for (int i = 0; i < 19; i++)
    {
        game.step(action, observation);
    }

    REQUIRE(game.missing_steps == 0);
}
