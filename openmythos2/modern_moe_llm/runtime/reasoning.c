#include <stdio.h>
#include <string.h>
#include "agent.h"

void agent_init(AgentContext *ctx) {
    ctx->state = STATE_IDLE;
    ctx->num_tools = 0;
    ctx->tools = NULL;
    memset(ctx->current_buffer, 0, sizeof(ctx->current_buffer));
}

char* agent_execute_tool(AgentContext *ctx, const char* tool_name, const char* args) {
    printf("[Skill/Tool Call] Solicitando %s com argumentos: %s\n", tool_name, args);
    // Loop de skills para invocar a correspondente
    return "{ \"status\": \"success\", \"result\": \"mocked_output\" }"; 
}

// Em tempo real, cada token gerado passa por aqui para capturar o fluxo de pensamento do LLM
void agent_process_token(AgentContext *ctx, const char* token_str, Model *model) {
    strcat(ctx->current_buffer, token_str);
    
    // Transições Híbridas ReAct/Plan
    if (strstr(ctx->current_buffer, "<plan>")) {
        ctx->state = STATE_PLANNING;
        printf("\n\033[36m[Thinking Engine] Iniciando planejamento multi-etapa...\033[0m\n");
        memset(ctx->current_buffer, 0, sizeof(ctx->current_buffer));
    } 
    else if (strstr(ctx->current_buffer, "<tool_call>")) {
        // Encontramos o comando de chamar uma tool...
        ctx->state = STATE_EXECUTING_TOOL;
        // Simulando que extraímos o json e identificamos
        char* obs = agent_execute_tool(ctx, "web_search", "{\"query\": \"latest llm papers\"}"); // mock run
        printf("\033[32m[Observation] Inserindo retorno no KV Cache do modelo...\033[0m\n");
        // Em prod: tokenizar o "obs" e injetar no KV cache via model_forward()
        ctx->state = STATE_INTEGRATING_OBSERVATION;
        memset(ctx->current_buffer, 0, sizeof(ctx->current_buffer));
    }
}
