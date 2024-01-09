#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../src/instructions.h"

TEST_CASE("Instruction and RecoverableInstruction Size Test")
{
    REQUIRE(sizeof(Instruction) == sizeof(RecoverableInstruction));
}

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

TEST_CASE("ConvertToRecoverable Complex Function Test")
{
    // Create a mock datastream and weights array
    float datastream[30];
    float weights[30];

    // Create a vector of Instructions
    vector<Instruction> instructions = {
        Instruction(COPY, 5, 5, &datastream[0], &datastream[5]),
        Instruction(DOT, 5, 5, &datastream[0], &datastream[5], &weights[0]),
        Instruction(ELEMENTWISE_MULTIPLY, 5, 5, &datastream[5], &datastream[10]),
        Instruction(ELEMENTWISE_ADD, 5, 5, &datastream[10], &datastream[15]),
        Instruction(SCALER_ADD, 5, 5, 5.0f, &datastream[15], &datastream[20], nullptr),
        Instruction(SCALER_MULTIPLY, 5, 5, -1.0f, &datastream[20], &datastream[25], nullptr)};

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
        if (instructions[i].op == SCALER_ADD)
        {
            REQUIRE(recoverableInstructions[i].addr3 == 0xffffffffffff);
            REQUIRE(recoverableInstructions[i].scalar1 == instructions[i].scalar1);
            REQUIRE(recoverableInstructions[i].scalar1 == 5.0f);
        }
        else if (instructions[i].op == SCALER_MULTIPLY)
        {
            REQUIRE(recoverableInstructions[i].addr3 == 0xffffffffffff);
            REQUIRE(recoverableInstructions[i].scalar1 == instructions[i].scalar1);
            REQUIRE(recoverableInstructions[i].scalar1 == -1.0f);
        }
        else if (instructions[i].op == DOT)
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

TEST_CASE("Practical -> Recoverable -> Practical Conversion Test")
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

TEST_CASE("Practical -> Recoverable -> Practical Complex Function Test")
{
    // Create a mock datastream and weights array
    float datastream[30];
    float weights[30];

    // Create a vector of Instructions
    vector<Instruction> instructions = {
        Instruction(COPY, 5, 5, &datastream[0], &datastream[5]),
        Instruction(DOT, 5, 5, &datastream[0], &datastream[5], &weights[0]),
        Instruction(ELEMENTWISE_MULTIPLY, 5, 5, &datastream[5], &datastream[10]),
        Instruction(ELEMENTWISE_ADD, 5, 5, &datastream[10], &datastream[15]),
        Instruction(SCALER_ADD, 5, 5, 5.0f, &datastream[15], &datastream[20], nullptr),
        Instruction(SCALER_MULTIPLY, 5, 5, -1.0f, &datastream[20], &datastream[25], nullptr)};

    // Convert to RecoverableInstructions
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

TEST_CASE("Recoverable -> Practical -> Recoverable Conversion Test")
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

TEST_CASE("Recoverable -> Practical -> Recoverable Complex Function Test")
{
    // Create a mock datastream and weights array
    float datastream[30];
    float weights[30];

    // Create a vector of Instructions
    vector<RecoverableInstruction> recoverable = {
        RecoverableInstruction(COPY, 5, 5, 0, 5),
        RecoverableInstruction(DOT, 5, 5, 0, 5, 0),
        RecoverableInstruction(ELEMENTWISE_MULTIPLY, 5, 5, 5, 10),
        RecoverableInstruction(ELEMENTWISE_ADD, 5, 5, 10, 15),
        RecoverableInstruction(SCALER_ADD, 5, 5, 5.0f, 15, 20, 0xffffffffffff),
        RecoverableInstruction(SCALER_MULTIPLY, 5, 5, -1.0f, 20, 25, 0xffffffffffff)};

    // Convert to RecoverableInstructions
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
