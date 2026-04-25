#include <stdio.h>
#include <string.h>
#include "llm.h"
#include "agent.h"

int main(int argc, char **argv) {
    printf("==========================================\n");
    printf("   MODERN MoE LLM - NATIVE C RUNTIME\n");
    printf("   [MoE] [FlashAttn] [FP16 / KV Q2_K_TURBO]\n");
    printf("==========================================\n");
    
    // Init Memory Subsystem (SSD Paging)
    extern void memory_manager_init();
    memory_manager_init();

    // Init Model Architecture
    Model *llm = load_gguf_model("modern_moe_fp16.gguf");
    
    // ConfigAgent
    AgentContext sys_agent;
    agent_init(&sys_agent);
    
    printf("\nPronto para interacao (Modo Simulação).\n");
    
    // Loop principal (Mock)
    const char* mock_response[] = {
        "Olá! ", "<plan>", " Entendido o objetivo.", "</plan>", 
        " Deixe-me ver. ", "<tool_call>{\"tool\": \"web\"}</tool_call>", 
        "A resposta é 42."
    };
    
    for (int i=0; i<7; i++) {
        // model_forward() seria chamado aqui para emitir o próximo token.
        // Simulando a saída do loop autoregressivo para o Thinking Engine
        const char *token = mock_response[i];
        printf("%s", token);
        fflush(stdout);
        agent_process_token(&sys_agent, token, llm);
    }
    printf("\n");
    
    return 0;
}
