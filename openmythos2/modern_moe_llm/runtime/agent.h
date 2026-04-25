#ifndef AGENT_H
#define AGENT_H

#include "llm.h"

typedef enum {
    STATE_IDLE,
    STATE_PLANNING,
    STATE_EXECUTING_TOOL,
    STATE_INTEGRATING_OBSERVATION,
    STATE_GENERATING_RESPONSE
} ReasoningState;

// Interface simplificada de ferramenta
typedef struct {
    char name[64];
    char description[256];
    char* (*execute)(const char* json_args);
} Tool;

typedef struct {
    ReasoningState state;
    Tool *tools;
    int num_tools;
    char current_buffer[4096];
} AgentContext;

void agent_init(AgentContext *ctx);
void register_tool(AgentContext *ctx, Tool t);

// Ciclo do ReAct / Thinking
void agent_process_token(AgentContext *ctx, const char* token_str, Model *model);
char* agent_execute_tool(AgentContext *ctx, const char* tool_name, const char* args);

#endif // AGENT_H
