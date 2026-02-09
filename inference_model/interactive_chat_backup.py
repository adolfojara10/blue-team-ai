import sys
import os
import argparse
import logging
import pandas as pd
import threading
import time
from datetime import datetime

# Adjust path for local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_filter.llm_classifier import Llama3Classifier
from inference_model.inference import OllamaChatClient
from input.ensemble import ParallelEnsembleClassifier


class InteractiveChatbot:
    def __init__(self, logger, ollama_chat: OllamaChatClient, ensemble: ParallelEnsembleClassifier = None):
        self.logger = logger
        self.llama3 = Llama3Classifier(logger=logger, model="llama-guard3:1b")
        self.ollama_chat = ollama_chat
        self.ensemble = ensemble
        self.conversation_history = []
        self.session_start = datetime.now()
        
    def display_welcome_message(self):
        print("\n" + "="*60)
        print("🤖 INTERACTIVE AI CHATBOT")
        print("="*60)
        print("🔒 Safety-filtered conversations with AI")
        print("💬 Type your message and press Enter to chat")
        print("📝 Commands:")
        print("   • 'quit', 'exit', 'bye' - End conversation")
        print("   • 'history' - Show conversation history")
        print("   • 'clear' - Clear conversation history")
        print("   • 'help' - Show this help message")
        print("="*60)
        print(f"🕐 Session started: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print()

    def display_help(self):
        print("\n📖 HELP - Available Commands:")
        print("   • Normal text - Chat with the AI")
        print("   • 'quit'/'exit'/'bye' - End the conversation")
        print("   • 'history' - View all messages in this session")
        print("   • 'clear' - Clear conversation history")
        print("   • 'help' - Show this help message")
        print()

    def show_conversation_history(self):
        if not self.conversation_history:
            print("📝 No conversation history yet. Start chatting!")
            return
            
        print(f"\n📚 CONVERSATION HISTORY ({len(self.conversation_history)} messages)")
        print("-" * 50)
        for i, entry in enumerate(self.conversation_history, 1):
            timestamp = entry['timestamp'].strftime('%H:%M:%S')
            print(f"[{timestamp}] 👤 You: {entry['user_message']}")
            if entry['bot_response']:
                print(f"[{timestamp}] 🤖 Bot: {entry['bot_response']}")
            else:
                print(f"[{timestamp}] ⚠️  Bot: [Message blocked for safety]")
            if i < len(self.conversation_history):
                print()
        print("-" * 50)

    def clear_history(self):
        self.conversation_history.clear()
        print("🗑️  Conversation history cleared!")

    def is_safe_prediction(self, prediction):
        """Check if prediction indicates safe content"""
        return (isinstance(prediction, str) and "Safe" in prediction) or prediction == 0

    def classify_and_respond(self, user_input):
        """Main logic for classifying user input and generating response"""
        try:
            # Step 1: Classify the user prompt
            self.logger.info(f"🔍 Classifying user input: {user_input}")
            prediction, probabilities, metadata = self.ensemble.predict_sync(user_input)
            
            # Determine final prediction (with Llama3 for close decisions)
            if abs(metadata['vote_counts'][0] - metadata['vote_counts'][1]) <= 2:
                final_prediction = metadata['llama3_prediction']
                decision_method = "Llama3 (close decision)"
            else:
                final_prediction = prediction
                decision_method = "Ensemble consensus"
                
            self.logger.info(f"🧠 Classification result: {final_prediction} (via {decision_method})")

            # Step 2: If input is safe, generate LLM response
            if self.is_safe_prediction(final_prediction):
                self.logger.info("✅ Input classified as SAFE. Generating response...")
                
                # Generate response from LLM
                llm_response = self.ollama_chat.send_prompt(user_input)
                self.logger.info(f"🔍 Response: {llm_response}")

                # Step 3: Classify the LLM response for safety
                output_classification = self.llama3.classify_response(llm_response)
                self.logger.info(f"🔍 Response classification: {output_classification}")
                
                # Step 4: Return response if safe, otherwise block it
                if self.is_safe_prediction(output_classification):
                    self.logger.info("✅ Response classified as SAFE.")
                    return {
                        'success': True,
                        'response': llm_response,
                        'input_safety': final_prediction,
                        'output_safety': output_classification,
                        'metadata': metadata
                    }
                else:
                    self.logger.warning("⚠️ Response classified as UNSAFE. Blocking response.")
                    return {
                        'success': False,
                        'response': "I apologize, but I cannot provide a response to that query as it may contain unsafe content.",
                        'input_safety': final_prediction,
                        'output_safety': output_classification,
                        'reason': 'Unsafe response generated'
                    }
            else:
                self.logger.warning("⚠️ Input classified as UNSAFE. Blocking request.")
                return {
                    'success': False,
                    'response': "I cannot respond to that request as it appears to contain unsafe or inappropriate content.",
                    'input_safety': final_prediction,
                    'output_safety': None,
                    'reason': 'Unsafe input detected'
                }
                
        except Exception as e:
            self.logger.error(f"❌ Error in classify_and_respond: {e}", exc_info=True)
            return {
                'success': False,
                'response': "I'm sorry, I encountered an error processing your request. Please try again.",
                'error': str(e)
            }

    def add_to_history(self, user_message, bot_response, success=True):
        """Add conversation entry to history"""
        entry = {
            'timestamp': datetime.now(),
            'user_message': user_message,
            'bot_response': bot_response if success else None,
            'success': success
        }
        self.conversation_history.append(entry)

    def start_chat(self):
        """Main chat loop"""
        self.display_welcome_message()
        
        try:
            while True:
                # Get user input
                try:
                    user_input = input("👤 You: ").strip()
                except (KeyboardInterrupt, EOFError):
                    print("\n\n👋 Goodbye! Chat session ended.")
                    break

                if not user_input:
                    continue

                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Goodbye! Thanks for chatting!")
                    break
                elif user_input.lower() == 'history':
                    self.show_conversation_history()
                    continue
                elif user_input.lower() == 'clear':
                    self.clear_history()
                    continue
                elif user_input.lower() == 'help':
                    self.display_help()
                    continue

                # Show thinking indicator
                print("🤖 Bot: 🤔 Thinking...", end='', flush=True)
                
                # Process the message
                result = self.classify_and_respond(user_input)
                
                # Clear thinking indicator
                print("\r" + " " * 20 + "\r", end='')
                
                
                # Display response
                print(f"🤖 Bot: {result['response']}")
                
                # Add to history
                self.add_to_history(user_input, result['response'], result['success'])
                
                # Optional: Show safety info in debug mode
                if not result['success']:
                    print(f"   🔒 Safety Note: {result.get('reason', 'Content filtered')}")
                
                print()  # Add spacing between messages

        except Exception as e:
            self.logger.error(f"❌ Chat session error: {e}", exc_info=True)
            print(f"\n❌ An error occurred: {e}")
        
        finally:
            # Session summary
            session_duration = datetime.now() - self.session_start
            print(f"\n📊 Session Summary:")
            print(f"   ⏱️  Duration: {session_duration}")
            print(f"   💬 Messages exchanged: {len(self.conversation_history)}")
            print(f"   📝 Log file: Check your logs for detailed information")