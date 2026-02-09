# File: /final-filter/final-filter/src/inference_model/interactive_chat.py
import time


class InteractiveChatbot:
    """Interactive chatbot with safety filtering, CSV logging, and resource monitoring"""
    
    def __init__(self, logger, ollama_chat, ensemble, csv_logger=None):
        self.logger = logger
        self.ollama_chat = ollama_chat
        self.ensemble = ensemble
        self.csv_logger = csv_logger
        self.conversation_history = []
        self.show_resource_stats = True  # Toggle for showing resource stats
    
    def start_chat(self):
        """Start the interactive chat session"""
        print("\n" + "=" * 60)
        print("🤖 Welcome to the Safe AI Chatbot!")
        print("=" * 60)
        print("Commands:")
        print("  'quit' or 'exit' - End the conversation")
        print("  'clear' - Clear conversation history")
        print("  'stats' - Toggle resource statistics display")
        print("=" * 60 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit']:
                    print("\n👋 Goodbye! Stay safe!")
                    break
                
                if user_input.lower() == 'clear':
                    self.conversation_history = []
                    print("🗑️ Conversation history cleared.\n")
                    continue
                
                if user_input.lower() == 'stats':
                    self.show_resource_stats = not self.show_resource_stats
                    status = "enabled" if self.show_resource_stats else "disabled"
                    print(f"📊 Resource statistics display {status}.\n")
                    continue
                
                # Process the input
                self._process_input(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Stay safe!")
                break
            except Exception as e:
                self.logger.error(f"Error in chat: {e}")
                print(f"❌ An error occurred: {e}\n")
    
    def _process_input(self, user_input: str):
        """Process user input with safety filtering and resource monitoring"""
        llmguard_input_result = None
        llmguard_output_result = None
        llmguard_time = 0.0
        llm_response = None
        
        try:
            # Get safety prediction (now includes resource_stats)
            prediction, probabilities, metadata, model_results, resource_stats = self.ensemble.predict_sync(user_input)
            
            # Check if LLM Guard was called
            if metadata.get('llama3_prediction'):
                llmguard_input_result = metadata['llama3_prediction']
                llmguard_time = metadata.get('llama3_time', 0.0)
            
            # Determine safety
            is_safe = False
            if metadata.get('llama3_prediction'):
                if "safe" in str(metadata['llama3_prediction']).lower() and \
                   "unsafe" not in str(metadata['llama3_prediction']).lower():
                    is_safe = True
            else:
                is_safe = (prediction == 0)
            
            if is_safe:
                # Get response from LLM
                llm_response = self.ollama_chat.send_prompt(user_input)
                
                # Check output safety
                output_start = time.time()
                llmguard_output_result = self.ensemble.llama3.classify_response(llm_response)
                llmguard_time += time.time() - output_start
                
                # Check if output is safe
                output_safe = "safe" in str(llmguard_output_result).lower() and \
                              "unsafe" not in str(llmguard_output_result).lower()
                
                if output_safe:
                    print(f"\n🤖 Assistant: {llm_response}")
                else:
                    print(f"\n⚠️ The response was flagged as potentially unsafe.")
                    print(f"   Reason: {llmguard_output_result}")
            else:
                print(f"\n⚠️ Your message was flagged as potentially unsafe.")
                print(f"   Please rephrase your question.")
            
            # Show resource stats if enabled
            if self.show_resource_stats and resource_stats:
                print(f"\n   📊 Resources: CPU {resource_stats.get('cpu_percent_avg', 0):.1f}% | ", end="")
                if resource_stats.get('gpu_available'):
                    print(f"GPU {resource_stats.get('gpu_percent_avg', 0):.1f}% | ", end="")
                print(f"Time {metadata['total_time']:.2f}s")
            
            print()  # Empty line for readability
            
            # Log to CSV
            if self.csv_logger:
                self.csv_logger.log_prediction(
                    prompt=user_input,
                    final_prediction=prediction,
                    final_probabilities=probabilities.tolist(),
                    metadata=metadata,
                    model_results=model_results,
                    llmguard_input_result=llmguard_input_result,
                    llmguard_output_result=llmguard_output_result,
                    llmguard_time=llmguard_time,
                    llm_response=llm_response,
                    resource_stats=resource_stats
                )
                
        except Exception as e:
            self.logger.error(f"Error processing input: {e}")
            print(f"❌ Error processing your message: {e}\n")