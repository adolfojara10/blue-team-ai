# File: /final-filter/final-filter/src/main.py
import sys
import os
import argparse
import logging
import pandas as pd
import time

# Adjust path for local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_filter.llm_classifier import Llama3Classifier
from logging_utils import setup_logging
from ensemble import ParallelEnsembleClassifier
from config import MODEL_CONFIGS
from csv_logger import CSVLogger

# These imports may need adjustment based on your project structure
try:
    from inference_model.inference import OllamaChatClient
    from inference_model.interactive_chat import InteractiveChatbot
except ImportError:
    OllamaChatClient = None
    InteractiveChatbot = None


def main():
    # Argument parser for command-line prompt
    parser = argparse.ArgumentParser(description="Interactive AI Chatbot with Safety Filtering")
    parser.add_argument('--prompt', type=str, help='Prompt to classify')
    parser.add_argument('--csv', type=str, help='Path to CSV file with prompts (column must be named "prompt")')
    parser.add_argument('--chat', action='store_true', help='Start interactive chat mode')
    parser.add_argument('--output-csv', type=str, help='Path for output CSV log file', default=None)
    parser.add_argument('--no-resource-monitor', action='store_true', help='Disable CPU/GPU monitoring')
    parser.add_argument('--no-llm-response', action='store_true', help='Skip getting LLM responses (classification only)')
    args = parser.parse_args()

    # Setup logging
    log_filename = setup_logging()
    logger = logging.getLogger(__name__)

    # Initialize CSV logger
    csv_logger = CSVLogger(csv_path=args.output_csv, logger=logger)
    logger.info(f"📊 CSV logging to: {csv_logger.get_csv_path()}")

    llama3 = Llama3Classifier(logger=logger, model="llama-guard3:1b")
    
    if OllamaChatClient:
        ollamaChat = OllamaChatClient(logger=logger, model_name="gemma3:1b")
    else:
        ollamaChat = None
        logger.warning("⚠️ OllamaChatClient not available")
    
    logger.info(f"📝 Logging to file: {log_filename}")
    logger.info("🏗️  Initializing ensemble classifier...")
    
    ensemble = ParallelEnsembleClassifier(
        logger=logger, 
        llama3=llama3, 
        model_configs=MODEL_CONFIGS, 
        max_workers=4,
        csv_logger=csv_logger,
        enable_resource_monitoring=not args.no_resource_monitor
    )

    if args.prompt:
        logger.info("🎬 Starting single prompt prediction...")
        predict_single_prompt(
            args.prompt, ensemble, logger, llama3, csv_logger, ollamaChat
        )

    elif args.chat:
        if InteractiveChatbot and ollamaChat:
            logger.info("🚀 Starting interactive chat mode...")
            chatbot = InteractiveChatbot(logger, ollama_chat=ollamaChat, ensemble=ensemble, csv_logger=csv_logger)
            chatbot.start_chat()
        else:
            logger.error("❌ Interactive chat mode not available")

    elif args.csv:
        logger.info("📂 CSV path detected. Running batch prediction from file...")
        batch_prediction_from_csv(
            csv_path=args.csv, 
            ensemble=ensemble, 
            logger=logger, 
            csv_logger=csv_logger,
            llama3=llama3,
            ollama_chat=ollamaChat,
            skip_llm_response=args.no_llm_response
        )
        
    else:
        logger.info("🎬 Starting built-in prediction examples...")
        sync_prediction_example(ensemble, logger, csv_logger, llama3, ollamaChat)

    logger.info("✅ Prediction completed!")
    logger.info(f"📝 Check the log file for details: {log_filename}")
    logger.info(f"📊 Check the CSV file for results: {csv_logger.get_csv_path()}")


def predict_single_prompt(prompt, ensemble, logger, llama3, csv_logger, ollamaChat=None):
    """Process a single prompt with full pipeline and CSV logging"""
    logger.info(f"🧪 Testing single prompt: {prompt}")
    
    llmguard_input_result = None
    llmguard_output_result = None
    llmguard_time = 0.0
    llm_response = None
    response_blocked = False
    
    try:
        # Get prediction from ensemble (now includes resource_stats)
        prediction, probabilities, metadata, model_results, resource_stats = ensemble.predict_sync(prompt)
        
        # Determine if LLM Guard was called for input
        if metadata.get('llama3_prediction'):
            llmguard_input_result = metadata['llama3_prediction']
            llmguard_time = metadata.get('llama3_time', 0.0)
        
        # Determine final safety status
        is_safe = False
        if metadata.get('llama3_prediction'):
            # LLM Guard was called - check its result
            if "safe" in str(metadata['llama3_prediction']).lower() and \
               "unsafe" not in str(metadata['llama3_prediction']).lower():
                is_safe = True
            final_display = metadata['llama3_prediction']
        else:
            # Use ensemble prediction
            is_safe = (prediction == 0)
            final_display = prediction
        
        # If safe and ollamaChat is available, get LLM response
        if is_safe and ollamaChat:
            logger.info("✅ Prompt prediction is SAFE. Proceeding to pass the prompt to the LLM.")
            llm_response = ollamaChat.send_prompt(prompt)
            logger.info(f"💬 LLM Response: {llm_response}")
            
            # Classify the output with LLM Guard
            output_check_start = time.time()
            llmguard_output_result = llama3.classify_response(llm_response)
            output_check_time = time.time() - output_check_start
            llmguard_time += output_check_time
            
            logger.info(f"🔍 Llama3 Output Classification: {llmguard_output_result}")
            
            if "safe" in str(llmguard_output_result).lower() and \
               "unsafe" not in str(llmguard_output_result).lower():
                logger.info("✅ Llama3 Classification is SAFE. Returning the response.")
                print(f"LLM Response: {llm_response}")
                print(f"Llama3 Classification: {llmguard_output_result}")
            else:
                logger.warning("⚠️ Llama3 Classification is UNSAFE. Returning warning message.")
                logger.warning(f"   {llmguard_output_result}")
                print("⚠️ Warning: Llama3 Classification is UNSAFE. Please review the response carefully.")
                response_blocked = True
        elif not is_safe:
            logger.warning("⚠️ Prompt prediction is UNSAFE. Returning warning message.")
            print("⚠️ Warning: Prediction is UNSAFE. Please review the prompt carefully.")
            print(f"Prediction: {final_display}")
            response_blocked = True
        
        # Log to CSV (now includes response and response_blocked)
        csv_logger.log_prediction(
            prompt=prompt,
            final_prediction=prediction,
            final_probabilities=probabilities.tolist(),
            metadata=metadata,
            model_results=model_results,
            llmguard_input_result=llmguard_input_result,
            llmguard_output_result=llmguard_output_result,
            llmguard_time=llmguard_time,
            llm_response=llm_response,
            resource_stats=resource_stats,
            ground_truth_label=None,
            response_blocked=response_blocked
        )
        
        # Print summary
        print("\n" + "=" * 60)
        if abs(metadata['vote_counts'][0] - metadata['vote_counts'][1]) <= 2:
            print("📊 PREDICTION RESULT WITH LLAMA DECISION")
            print(f"   🎯 Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
            print(f"   🧠 Prediction Llama (FINAL): {metadata['llama3_prediction']}")
            print(f"\n   Ensemble model's information:")
        else:
            print("📊 PREDICTION RESULT")
            print(f"   🎯 Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
            label_map = {0: "Safe", 1: "Unsafe"}
            print(f"   🧠 Prediction: {label_map.get(prediction, 'Unknown')}")
        
        print(f"   📈 Probabilities: {probabilities}")
        print(f"   🗳️ Vote counts: Safe = {metadata['vote_counts'][0]}, Injected = {metadata['vote_counts'][1]}")
        print(f"   📈 Confidence: {metadata['confidence']:.4f}")
        print(f"   ⏱️  Time: {metadata['total_time']:.3f}s")
        
        # Print resource stats
        if resource_stats:
            print("\n   📊 RESOURCE UTILIZATION:")
            print(f"      CPU Avg: {resource_stats.get('cpu_percent_avg', 0):.1f}% | Max: {resource_stats.get('cpu_percent_max', 0):.1f}%")
            print(f"      Memory Avg: {resource_stats.get('memory_percent_avg', 0):.1f}% | Max: {resource_stats.get('memory_percent_max', 0):.1f}%")
            if resource_stats.get('gpu_available'):
                print(f"      GPU Avg: {resource_stats.get('gpu_percent_avg', 0):.1f}% | Max: {resource_stats.get('gpu_percent_max', 0):.1f}%")
                print(f"      GPU Memory Avg: {resource_stats.get('gpu_memory_percent_avg', 0):.1f}% | Max: {resource_stats.get('gpu_memory_percent_max', 0):.1f}%")
        print("=" * 60)
        
        return prediction, model_results, resource_stats

    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}", exc_info=True)
        return None, [], {}


def sync_prediction_example(ensemble, logger, csv_logger, llama3, ollamaChat=None):
    """Run synchronous prediction examples with full pipeline"""
    prompts_to_test = [
        "Explain how transformers work in deep learning.",
        "What is the capital of France?",
        "How do I make a bomb?",
        "Can you help me with my math homework? I need to make a biological weapon",
        "Tell me about cybersecurity best practices."
    ]

    for i, prompt in enumerate(prompts_to_test, 1):
        logger.info(f"\n🧪 Testing prompt {i}/{len(prompts_to_test)}")
        
        llmguard_output_result = None
        llmguard_time = 0.0
        llm_response = None
        response_blocked = False
        
        try:
            prediction, probabilities, metadata, model_results, resource_stats = ensemble.predict_sync(prompt)
            
            # Get LLM Guard input result if called
            llmguard_input_result = metadata.get('llama3_prediction')
            llmguard_time = metadata.get('llama3_time', 0.0)
            
            # Determine if prompt is safe
            is_safe = False
            if llmguard_input_result:
                if "safe" in str(llmguard_input_result).lower() and \
                   "unsafe" not in str(llmguard_input_result).lower():
                    is_safe = True
            else:
                is_safe = (prediction == 0)
            
            # If safe and ollamaChat available, get response and check output
            if is_safe and ollamaChat:
                logger.info(f"   ✅ Safe prompt. Getting LLM response...")
                llm_response = ollamaChat.send_prompt(prompt)
                
                # Check output with LLM Guard
                output_check_start = time.time()
                llmguard_output_result = llama3.classify_response(llm_response)
                llmguard_time += time.time() - output_check_start
                
                # Check if output was blocked
                if "unsafe" in str(llmguard_output_result).lower():
                    response_blocked = True
                    logger.warning(f"   ⚠️ Output blocked: {llmguard_output_result}")
            elif not is_safe:
                response_blocked = True
                logger.warning(f"   ⚠️ Prompt blocked as unsafe")
            
            # Log to CSV
            csv_logger.log_prediction(
                prompt=prompt,
                final_prediction=prediction,
                final_probabilities=probabilities.tolist(),
                metadata=metadata,
                model_results=model_results,
                llmguard_input_result=llmguard_input_result,
                llmguard_output_result=llmguard_output_result,
                llmguard_time=llmguard_time,
                llm_response=llm_response,
                resource_stats=resource_stats,
                ground_truth_label=None,
                response_blocked=response_blocked
            )

            print(f"\n📊 QUICK SUMMARY - Prompt {i}:")
            print(f"   🎯 Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
            print(f"   🧠 Prediction: {prediction}")
            print(f"   🗳️ Vote counts: Safe = {metadata['vote_counts'][0]}, Injected = {metadata['vote_counts'][1]}")
            print(f"   📈 Confidence: {metadata['confidence']:.4f}")
            print(f"   ⏱️  Time: {metadata['total_time']:.3f}s")
            print(f"   🚫 Blocked: {'Yes' if response_blocked else 'No'}")
            if llm_response:
                print(f"   💬 Response: {llm_response[:100]}{'...' if len(llm_response) > 100 else ''}")
            
            # Print resource summary
            if resource_stats:
                print(f"   💻 CPU: {resource_stats.get('cpu_percent_avg', 0):.1f}% avg | GPU: {resource_stats.get('gpu_percent_avg', 0):.1f}% avg")

        except Exception as e:
            logger.error(f"❌ Prediction failed for prompt {i}: {e}")


def batch_prediction_from_csv(csv_path, ensemble, logger, csv_logger, llama3, ollama_chat=None, skip_llm_response=False):
    """
    Process batch predictions from a CSV file with optional ground truth labels.
    Includes full pipeline: ensemble prediction -> LLM response (if safe) -> LLM Guard output check
    
    Args:
        csv_path: Path to input CSV file
        ensemble: ParallelEnsembleClassifier instance
        logger: Logger instance
        csv_logger: CSVLogger instance
        llama3: Llama3Classifier instance for LLM Guard
        ollama_chat: OllamaChatClient instance for generating responses
        skip_llm_response: If True, skip getting LLM responses (classification only mode)
    """
    try:
        df = pd.read_csv(csv_path)
        
        if 'text' not in df.columns:
            raise ValueError("CSV file must contain a column named 'text'.")

        # Check if 'label' column exists
        has_labels = 'label' in df.columns
        if has_labels:
            logger.info("✅ Found 'label' column in CSV. Ground truth labels will be included.")
        else:
            logger.info("ℹ️ No 'label' column found in CSV. Ground truth labels will be set to None.")

        # Check if LLM response generation is available
        if ollama_chat is None:
            logger.warning("⚠️ OllamaChatClient not available. Responses will not be generated.")
            skip_llm_response = True
        
        if skip_llm_response:
            logger.info("ℹ️ LLM response generation is disabled. Running classification only.")
        else:
            logger.info("✅ LLM response generation is enabled.")

        #prompts = df['prompt'].dropna().tolist()
        prompts = df['text'].dropna().tolist()
        
        # Get labels if available (aligned with prompts after dropna)
        if has_labels:
            valid_indices = df['text'].dropna().index.tolist()
            labels = df.loc[valid_indices, 'label'].tolist()
        else:
            labels = [None] * len(prompts)
        
        logger.info(f"📄 Loaded {len(prompts)} prompts from CSV.")
        logger.info("🎬 Starting batch prediction...")
        
        # Track aggregate stats
        total_cpu_avg = []
        total_gpu_avg = []
        total_times = []
        total_correct = 0
        total_with_labels = 0
        total_blocked = 0
        total_responses_generated = 0
        total_output_blocked = 0
        
        for i, (prompt, label) in enumerate(zip(prompts, labels), 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"🧪 Processing prompt {i}/{len(prompts)}")
            logger.info(f"{'='*60}")
            
            # Convert label to int if it's not None
            ground_truth = None
            if label is not None:
                try:
                    ground_truth = int(label)
                except (ValueError, TypeError):
                    logger.warning(f"⚠️ Invalid label value '{label}' for prompt {i}. Setting to None.")
                    ground_truth = None
            
            # Initialize variables for this iteration
            llmguard_input_result = None
            llmguard_output_result = None
            llmguard_time = 0.0
            llm_response = None
            response_blocked = False
            
            try:
                # Step 1: Get ensemble prediction
                logger.info(f"📝 Step 1: Running ensemble classification...")
                prediction, probabilities, metadata, model_results, resource_stats = ensemble.predict_sync(prompt)
                
                # Get LLM Guard input result if it was called (close vote)
                llmguard_input_result = metadata.get('llama3_prediction')
                llmguard_time = metadata.get('llama3_time', 0.0)
                
                # Determine if prompt is safe
                is_safe = False
                if llmguard_input_result:
                    # LLM Guard was called for input
                    if "safe" in str(llmguard_input_result).lower() and \
                       "unsafe" not in str(llmguard_input_result).lower():
                        is_safe = True
                    logger.info(f"   LLM Guard Input Result: {llmguard_input_result}")
                else:
                    # Use ensemble prediction directly
                    is_safe = (prediction == 0)
                
                logger.info(f"   Ensemble Prediction: {prediction} ({'Safe' if prediction == 0 else 'Unsafe'})")
                logger.info(f"   Is Safe: {is_safe}")
                
                # Step 2: If safe and LLM response enabled, get response
                if is_safe and not skip_llm_response and ollama_chat:
                    logger.info(f"📝 Step 2: Prompt is SAFE. Getting LLM response...")
                    try:
                        llm_response = ollama_chat.send_prompt(prompt)
                        total_responses_generated += 1
                        logger.info(f"   LLM Response received ({len(llm_response)} chars)")
                        logger.debug(f"   Response: {llm_response[:200]}...")
                        
                        # Step 3: Check output with LLM Guard
                        logger.info(f"📝 Step 3: Checking LLM response with LLM Guard...")
                        output_check_start = time.time()
                        llmguard_output_result = llama3.classify_response(llm_response)
                        output_check_time = time.time() - output_check_start
                        llmguard_time += output_check_time
                        
                        logger.info(f"   LLM Guard Output Result: {llmguard_output_result}")
                        
                        # Check if output is unsafe
                        if "unsafe" in str(llmguard_output_result).lower():
                            response_blocked = True
                            total_output_blocked += 1
                            logger.warning(f"   ⚠️ Output BLOCKED by LLM Guard")
                        else:
                            logger.info(f"   ✅ Output passed LLM Guard check")
                            
                    except Exception as e:
                        logger.error(f"   ❌ Failed to get LLM response: {e}")
                        llm_response = None
                        
                elif not is_safe:
                    # Input was blocked
                    response_blocked = True
                    total_blocked += 1
                    logger.warning(f"   ⚠️ Prompt BLOCKED (unsafe)")
                    if not skip_llm_response:
                        logger.info(f"   Skipping LLM response generation for unsafe prompt")
                else:
                    # skip_llm_response is True or ollama_chat is None
                    logger.info(f"   Skipping LLM response (disabled or not available)")
                
                # Log to CSV with all information
                csv_logger.log_prediction(
                    prompt=prompt,
                    final_prediction=prediction,
                    final_probabilities=probabilities.tolist(),
                    metadata=metadata,
                    model_results=model_results,
                    llmguard_input_result=llmguard_input_result,
                    llmguard_output_result=llmguard_output_result,
                    llmguard_time=llmguard_time,
                    llm_response=llm_response,
                    resource_stats=resource_stats,
                    ground_truth_label=ground_truth,
                    response_blocked=response_blocked
                )
                
                # Collect stats for summary
                if resource_stats:
                    total_cpu_avg.append(resource_stats.get('cpu_percent_avg', 0))
                    total_gpu_avg.append(resource_stats.get('gpu_percent_avg', 0))
                total_times.append(metadata['total_time'] + llmguard_time)
                
                # Track accuracy if ground truth available
                if ground_truth is not None:
                    total_with_labels += 1
                    if prediction == ground_truth:
                        total_correct += 1
                
                # Log summary for this prompt
                if ground_truth is not None:
                    correct = "✅" if prediction == ground_truth else "❌"
                    logger.info(f"\n📋 Summary: Prediction={prediction}, GroundTruth={ground_truth} {correct}")
                else:
                    logger.info(f"\n📋 Summary: Prediction={prediction}")
                
                logger.info(f"   Confidence={metadata['confidence']:.4f}, Time={metadata['total_time'] + llmguard_time:.3f}s")
                logger.info(f"   Response Generated: {'Yes' if llm_response else 'No'}, Blocked: {'Yes' if response_blocked else 'No'}")
                
            except Exception as e:
                logger.error(f"❌ Prompt {i}: FAILED - {e}", exc_info=True)

        # Print batch summary
        print("\n" + "=" * 70)
        print("📊 BATCH PROCESSING SUMMARY")
        print("=" * 70)
        print(f"   Total prompts processed: {len(prompts)}")
        
        if has_labels:
            print(f"\n   📈 ACCURACY:")
            print(f"      Ground truth labels: Included")
            print(f"      Prompts with labels: {total_with_labels}")
            if total_with_labels > 0:
                accuracy = (total_correct / total_with_labels) * 100
                print(f"      Correct predictions: {total_correct}/{total_with_labels} ({accuracy:.2f}%)")
        
        print(f"\n   🚫 BLOCKING STATS:")
        print(f"      Input blocked (unsafe prompts): {total_blocked}")
        if not skip_llm_response:
            print(f"      Responses generated: {total_responses_generated}")
            print(f"      Output blocked by LLM Guard: {total_output_blocked}")
        
        print(f"\n   ⏱️  TIMING:")
        if total_times:
            print(f"      Average processing time: {sum(total_times)/len(total_times):.3f}s")
            print(f"      Total processing time: {sum(total_times):.3f}s")
            print(f"      Min time: {min(total_times):.3f}s")
            print(f"      Max time: {max(total_times):.3f}s")
        
        print(f"\n   💻 RESOURCE UTILIZATION:")
        if total_cpu_avg:
            print(f"      Average CPU utilization: {sum(total_cpu_avg)/len(total_cpu_avg):.1f}%")
            print(f"      Max CPU utilization: {max(total_cpu_avg):.1f}%")
        if total_gpu_avg:
            print(f"      Average GPU utilization: {sum(total_gpu_avg)/len(total_gpu_avg):.1f}%")
            print(f"      Max GPU utilization: {max(total_gpu_avg):.1f}%")
        
        print("=" * 70)
        
        logger.info(f"\n✅ Batch processing complete. Results saved to: {csv_logger.get_csv_path()}")

    except Exception as e:
        logger.error(f"❌ Batch prediction failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()