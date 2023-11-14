#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../src/instructions.h"

TEST_CASE("ConvertToRecoverable Function Test")
{
    // Create a mock datastream and weights array
    float datastream[10];
    float weights[10];

    // Create a vector of Instructions
    vector<Instruction> instructions = {
        Instruction(COPY, 5, 5, &datastream[0], &datastream[5]),
        Instruction(DOT, 5, 5, &datastream[0], &datastream[5], &weights[0])};

    // Convert to RecoverableInstructions
    vector<RecoverableInstruction> recoverableInstructions = ConvertToRecoverable(instructions, datastream, weights);

    REQUIRE(recoverableInstructions.size() == instructions.size());
    for (size_t i = 0; i < instructions.size(); ++i)
    {
        REQUIRE(recoverableInstructions[i].op == instructions[i].op);
        REQUIRE(recoverableInstructions[i].size_in == instructions[i].size_in);
        REQUIRE(recoverableInstructions[i].size_out == instructions[i].size_out);
        REQUIRE(recoverableInstructions[i].addr1 == (instructions[i].addr1 - datastream));
        REQUIRE(recoverableInstructions[i].addr2 == (instructions[i].addr2 - datastream));
        if (instructions[i].op == DOT)
        {
            REQUIRE(recoverableInstructions[i].addr3 == (instructions[i].addr3 - weights));
        }
        else if (recoverableInstructions[i].addr3 == 0xffffffffffff)
        {
            REQUIRE(instructions[i].addr3 == nullptr);
        }
        else
        {
            REQUIRE(recoverableInstructions[i].addr3 == (instructions[i].addr3 - datastream));
        }
    }
}

TEST_CASE("Practical -> Recoverable -> Practical Conversion")
{
    float datastream[10];
    float weights[10];
    vector<Instruction> instructions = {
        Instruction(COPY, 5, 5, &datastream[0], &datastream[5]),
        Instruction(DOT, 5, 5, &datastream[0], &datastream[5], &weights[0])};

    auto recoverable = ConvertToRecoverable(instructions, datastream, weights);
    auto practicalBack = ConvertToPractical(recoverable, datastream, weights);

    REQUIRE(practicalBack.size() == instructions.size());
    for (size_t i = 0; i < instructions.size(); ++i)
    {
        REQUIRE(practicalBack[i].op == instructions[i].op);
        REQUIRE(practicalBack[i].size_in == instructions[i].size_in);
        REQUIRE(practicalBack[i].size_out == instructions[i].size_out);
        REQUIRE(practicalBack[i].addr1 == instructions[i].addr1);
        REQUIRE(practicalBack[i].addr2 == instructions[i].addr2);
        REQUIRE(practicalBack[i].addr3 == instructions[i].addr3);
    }
}

TEST_CASE("Recoverable -> Practical -> Recoverable Conversion")
{
    float datastream[10];
    float weights[10];
    vector<RecoverableInstruction> recoverable = {
        RecoverableInstruction(COPY, 5, 5, 0, 5, 0),
        RecoverableInstruction(DOT, 5, 5, 0, 5, 0)};

    auto practical = ConvertToPractical(recoverable, datastream, weights);
    auto recoverableBack = ConvertToRecoverable(practical, datastream, weights);

    REQUIRE(recoverableBack.size() == recoverable.size());
    for (size_t i = 0; i < recoverable.size(); ++i)
    {
        REQUIRE(recoverableBack[i].op == recoverable[i].op);
        REQUIRE(recoverableBack[i].size_in == recoverable[i].size_in);
        REQUIRE(recoverableBack[i].size_out == recoverable[i].size_out);
        REQUIRE(recoverableBack[i].addr1 == recoverable[i].addr1);
        REQUIRE(recoverableBack[i].addr2 == recoverable[i].addr2);
        REQUIRE(recoverableBack[i].addr3 == recoverable[i].addr3);
    }
}
