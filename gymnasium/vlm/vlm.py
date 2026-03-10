import torch
import openai
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from abc import ABC, abstractmethod
import re

class BaseVLM(ABC):
    """Base class for Vision-Language Models."""
    
    def __init__(self):
        """Initialize the VLM."""
        self.conversation_history = []
        
    def reset_conversation(self):
        """Reset the conversation history."""
        self.conversation_history = []

    def prune_conversation_history(self, num_interaction_history: int, instructions: str):
        """Prune the conversation history."""
        if num_interaction_history == 0:
            self.reset_conversation()
            return True
        # if the history is shorter than num_interaction_history*2, do nothing
        if len(self.conversation_history) < num_interaction_history * 2:
            return False
        # note that it needs to keep as pairs of user and assistant messages
        self.conversation_history = self.conversation_history[-num_interaction_history*2:]
        # make sure earliest message comes from user
        assert self.conversation_history[0]["role"] == "user" 
        # add instructions to the text part of the user message to its existing text
        if instructions not in self.conversation_history[0]["content"][1]["text"]:
            self.conversation_history[0]["content"][1]["text"] = re.sub(r'(?s)Environment feedback:.*?(?=This is step)', '', self.conversation_history[0]["content"][1]["text"])
            self.conversation_history[0]["content"][1]["text"] = instructions + self.conversation_history[0]["content"][1]["text"]
        return True
        
    @abstractmethod
    def query(self, image_base64: str, prompt: str) -> str:
        """
        Query the VLM with an image and a prompt.
        
        Args:
            image_base64: Base64 encoded image string
            prompt: The text prompt to send to the VLM
            
        Returns:
            The VLM's response as a string
        """
        pass
    


class TransformersVLM(BaseVLM):
    """VLM implementation using HuggingFace Transformers."""
    
    def __init__(self, model_name: str):
        """
        Initialize the Transformers VLM.
        
        Args:
            model_name: The name or path of the model to load
            device: The device to run the model on (default: "cuda")
        """
        super().__init__()
        run_name = model_name.split("/")[-2] # -1 is checkpoint num, -2 is run name
        if run_name == "mixed_all_v7_300k":
            model_cls = Qwen3VLForConditionalGeneration
        elif "qwen3" in run_name:
            model_cls = Qwen3VLForConditionalGeneration
        else:
            model_cls = Qwen2_5_VLForConditionalGeneration
        self.model = model_cls.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        
    def query(self, image_base64: str, prompt: str, final_goal_prompt: str = None, final_goal_img_b64: str = None, **kwargs) -> str:
        """
        Query the Transformers VLM with an image and a prompt.
        
        Args:
            image_base64: Base64 encoded image string
            prompt: The text prompt to send to the VLM
            kwargs: Additional keyword arguments
        Returns:
            The VLM's response as a string
        """
        # Build the user message
        user_message = {
            "role": "user",
            "content": [
                {"type": "image", "image": f"data:image;base64,{image_base64}"},
                {"type": "text", "text": prompt}
            ]
        }
        
        # Append to conversation history
        self.conversation_history.append(user_message)
        
        # Prepare input
        text = self.processor.apply_chat_template(
            self.conversation_history, 
            tokenize=False, 
            add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(self.conversation_history)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        
        # Generate response
        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        # Append assistant's response to conversation history
        assistant_message = {"role": "assistant", "content": output_text}
        self.conversation_history.append(assistant_message)
        
        return output_text

class OpenAIVLM(BaseVLM):
    """VLM implementation using OpenAI's API."""
    
    def __init__(self, api_key: str, model_name: str, base_url: str = None, max_completion_tokens: int = 8192):
        """
        Initialize the OpenAI VLM.
        
        Args:
            api_key: OpenAI API key
            model_name: The name of the OpenAI model to use
            max_completion_tokens: The maximum number of tokens to generate (default: 8192)
            client: The OpenAI client to use
        """
        super().__init__()
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.max_completion_tokens = max_completion_tokens
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.total_cost = 0
        self.provider = None
        if "qwen2.5-vl" in self.model_name:
            self.provider = {"order": ["together"], "allow_fallbacks": False}
        if "qwen-vl-max" in self.model_name:
            self.max_completion_tokens = 8200
        # if "qwen3-vl-235b-a22b-instruct" in self.model_name or "qwen2.5-vl-72b-instruct" in self.model_name:
        #     self.provider = "chutes/bf16"

        
    def query(self, image_base64: str, prompt: str, print_usage: bool = False, 
    final_goal_prompt: str = None, final_goal_img_b64: str = None) -> str:
        """
        Query the OpenAI VLM with an image and a prompt.
        
        Args:
            image_base64: Base64 encoded image string
            prompt: The text prompt to send to the VLM
            final_goal_prompt: The text prompt for the final goal
            final_goal_img_b64: The base64 encoded image string for the final goal
        Returns:
            The VLM's response as a string
        """
        
        # Build the user message
        user_message = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                {"type": "text", "text": prompt}
            ]
        }
        if final_goal_prompt is not None and final_goal_img_b64 is not None:
            user_message["content"].append({
                "type": "text", "text": final_goal_prompt
            })
            user_message["content"].append({
                "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{final_goal_img_b64}"}
            })
        # Append to conversation history
        self.conversation_history.append(user_message)
        max_trials = 0
        output_text = None
        response = None
        sleep_time = 1
        while max_trials < 10:
            # print(f"Conversation history: {self.conversation_history}")
            try:
                if self.provider is not None:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=self.conversation_history,
                        max_completion_tokens=self.max_completion_tokens,
                        extra_body={"provider": self.provider}
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=self.conversation_history,
                        max_completion_tokens=self.max_completion_tokens,
                        # extra_body={"usage": {"include": True}},   
                    )
                output_text = response.choices[0].message.content
                break  # Exit loop on successful response
            except Exception as e:
                print(f"Error: {e}")
                if "Total number of images exceeds" in str(e):
                    # truncate the conversation history
                    output_text = "Error: Total number of images exceeds the maximum context length"
                    break
                if "maximum context length is" in str(e):
                    output_text = "Error: Maximum context length is exceeded."
                    break
                max_trials += 1
                import time
                time.sleep(sleep_time)
                sleep_time *= 2
                print(f"Retrying..., max_trials: {max_trials}/10")
        
        if output_text is None:
            output_text = "Error in receiving a response after 10 trials"
        elif output_text == "":
            output_text = "Empty response"
    

        # optionally print the usage
        if print_usage and response is not None:
            self.total_cost += response.usage.cost
            print(f"Total cost: {self.total_cost}")
            
        # Append assistant's response to conversation history
        assistant_message = {"role": "assistant", "content": output_text}
        self.conversation_history.append(assistant_message)
        
        return output_text
    

    def query_text_mode(self, prompt: str, print_usage: bool = False) -> str:
        """
        Query the OpenAI VLM with an image and a prompt.
        
        Args:
            image_base64: Base64 encoded image string
            prompt: The text prompt to send to the VLM
            
        Returns:
            The VLM's response as a string
        """
        
        # Build the user message
        user_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ]
        }
        
        # Append to conversation history
        self.conversation_history.append(user_message)
        max_trials = 0
        output_text = None
        response = None
        while max_trials < 10:
            # print(f"Conversation history: {self.conversation_history}")
            try:
                if self.provider is not None:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=self.conversation_history,
                        max_completion_tokens=self.max_completion_tokens,
                        extra_body={"provider": self.provider}
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=self.conversation_history,
                        max_completion_tokens=self.max_completion_tokens,
                        # extra_body={"usage": {"include": True}},   
                    )
                output_text = response.choices[0].message.content
                break  # Exit loop on successful response
            except Exception as e:
                print(f"Error: {e}")
                max_trials += 1
                import time
                time.sleep(1)
                print(f"Retrying..., max_trials: {max_trials}/10")
        
        if output_text is None:
            output_text = "Error in receiving a response after 10 trials"
        elif output_text == "":
            output_text = "Empty response"
    

        # optionally print the usage
        if print_usage and response is not None:
            self.total_cost += response.usage.cost
            print(f"Total cost: {self.total_cost}")
            
        # Append assistant's response to conversation history
        assistant_message = {"role": "assistant", "content": output_text}
        self.conversation_history.append(assistant_message)
        
        return output_text
    