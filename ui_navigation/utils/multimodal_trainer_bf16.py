import os, json
from typing import List, Tuple, Dict, Any
import torch
from glob import glob
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW, SGD
from transformers import Trainer, TrainingArguments, get_scheduler
from models import MultiModalityCausalLM, VLChatProcessor
from utils.io import load_pil_images


system_desc_message = """You are a GUI agent.
I will give you a screenshot of a mobile phone.
**INSTRUCTION**: {instruction}

**TASK**: Given the screenshot and instruction, you should analyze the screen for relevant details that might pertain to the given query. This includes checking for specific applications, icons, or buttons that are visible, and any information or results that are currently displayed on the screen.
"""
system_act_message = """You are a GUI agent.
I will give you a screenshot of a mobile phone.
**INSTRUCTION**: {instruction}
**SCREEN DESCRIPTION**: {screen_desc}

**SCREEN COORDINATE SYSTEM**: A coordinate (x, y) represents a point on the screen. The first value, labeled as `x`, horizontal,i.e. x ranges from 0 to 1, meaning the position of point ranges from the left to right, where x<0.4 means left, 0.4<=x<=0.6 means middle and x>0.6 meansright. The second value, labeled as `y`, is vertical, i.e. y ranges from 0 to 1, meaning the position of point ranges from the bottom to top. where y<0.2means bottom, 0.2<=y<0.4 means lower, + 0.4<=y<0.5 means lower middle, 0.5<=y<=0.6 means upper middle, 0.6<y<=0.8 means upper, and y>0.8 means top. 
**TASK**: Given the screenshot and instruction, follow the bellow tasks to fulfill the instruction.
1. You should analyze the screen for relevant details that might pertain to the given query. This includes checking for specific applications, icons, or buttons that are visible, and any information or results that are currently displayed on the screen. The screen analysis should be like the **SCREEN DESCRIPTION** part.
2. After screen description through screen analysis, describe possible actions you may conduct.
3. Based on the possible actions conductable, you have to perform a final action on screen. You must answer by the following format: 'Action: result_action_type: ...,\nresult_action_text: ...,\nresult_touch_yx: ...,\nresult_lift_yx: ....'.
"""


class EnhancedMultiModalModel(MultiModalityCausalLM):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        image_token_masks=None,
        labels=None,
        **kwargs,
    ):
        """
        Forward method supporting joint image-text processing.
        """
        if pixel_values is not None and image_token_masks is not None:  # support both image inputs and masks
            inputs_embeds = self._process_multimodal_inputs(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_token_masks=image_token_masks,
            )
        else:                                                           # otherwise, use text embeddings only
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # Utilize already generated inputs_embeds
        kwargs.pop("inputs_embeds", None)
        outputs = self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
        return outputs

    def _process_multimodal_inputs(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_token_masks: torch.BoolTensor,
    ) -> torch.Tensor:
        """
        Generate combined embeddings from image and text tokens.

        Args:
            input_ids (torch.LongTensor): [batch_size, seq_length]
            pixel_values (torch.FloatTensor): [batch_size, num_images, 3, height, width]
            image_token_masks (torch.BoolTensor): [batch_size, seq_length]

        Returns:
            torch.Tensor: 결합된 토큰 임베딩 [batch_size, seq_length, hidden_dim]
        """
        bs, n = pixel_values.shape[0:2]
        images = pixel_values.view(bs * n, *pixel_values.shape[2:]) # flatten batch and image dimensions for vision model
        image_features = self.vision_model(images)                  # extract image features through vision model
        aligned_features = self.aligner(image_features)             # utilize alignment model

        aligned_features = aligned_features.view(bs, n, *aligned_features.shape[1:])    # [batch_size, num_images, token_length, hidden]
        aligned_features = aligned_features.flatten(1, 2)           # merge image tokens

        text_embeds = self.language_model.get_input_embeddings()(input_ids)             # get text embeddings
        aligned_features = aligned_features.to(text_embeds.dtype)

        for i in range(bs):
            num_image_tokens = image_token_masks[i].sum().item()    # insert aligned image embeddings into text embeddings
            num_aligned_tokens = aligned_features[i].shape[0]
            if num_image_tokens != num_aligned_tokens:
                raise ValueError(
                    f"Mismatch at sample {i}: "
                    f"image_token_masks has {num_image_tokens} tokens, "
                    f"but aligned_features has {num_aligned_tokens} tokens!"
                )
            text_embeds[i][image_token_masks[i]] = aligned_features[i]

        return text_embeds


class EnhancedMultiModalTrainer:
    def __init__(self, 
                 data_dir: str, 
                 pretrained_model_path: str, 
                 output_dir: str, 
                 batch_size: int = 2, 
                 max_epochs: int = 10, 
                 lr: float = 3e-4, 
                 user_question: str = system_act_message,
                 optimizer_name: str = "AdamW",
                 lora_config: dict = None,
                 training_args: dict = None):
        self.data_dir = data_dir
        self.pretrained_model_path = pretrained_model_path
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.lr = lr
        self.user_question = user_question
        self.optimizer_name = optimizer_name
        self.lora_config = lora_config if lora_config else {
            "r": 32,
            "lora_alpha": 64,
            "target_modules": ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
        self.training_args = training_args if training_args else {
            "bf16": True,
            "max_grad_norm": 1.0,
            "save_strategy": "epoch",
            "evaluation_strategy": "no",
            "logging_steps": 50,
            "save_total_limit": 2,
            "remove_unused_columns": False,
        }
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_data(self) -> List[Tuple[str, str]]:                  # load text-image pair
        self.samples = []
        episode_dirs = [                                            # list episode directories
            os.path.join(self.data_dir, d)
            for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d))
        ]
        for ep in episode_dirs:                                     # read each JSON file in episode folders
            for jp in glob(os.path.join(ep, "*.json")):
                with open(jp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                entries = data if isinstance(data, list) else [data]
                for ann in entries:
                    img_rel = ann["image_path"]                     # ann["image_path"] format : "general/...png"
                    prefix1  = "general" + os.sep
                    prefix2  = "google_apps" + os.sep
                    prefix3  = "install" + os.sep
                    prefix4  = "single" + os.sep
                    prefix5  = "web_shopping" + os.sep
                    if img_rel.startswith(prefix1):                 # remove redundant prefix(general) if present
                        img_rel = img_rel[len(prefix1):]
                    elif img_rel.startswith(prefix2):
                        img_rel = img_rel[len(prefix2):]
                    elif img_rel.startswith(prefix3):
                        img_rel = img_rel[len(prefix3):]
                    elif img_rel.startswith(prefix4):
                        img_rel = img_rel[len(prefix4):]
                    elif img_rel.startswith(prefix5):
                        img_rel = img_rel[len(prefix5):]
                    img_path = os.path.join(self.data_dir, img_rel)
                    self.samples.append((img_path, ann))
        pairs = []
        for i in range(self.__len__()):
            image_path, inst, screen_desc, action = self.__getitem__(i)
            pairs.append((image_path, inst, screen_desc, action))
        return pairs
    
    def __len__(self):
        # print("data num : ", len(self.samples))
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, ann = self.samples[idx]
        
        act_type = ""
        if ann.get("result_action_type", "") == 0:
            act_type = ""
        elif ann.get("result_action_type", "") == 1:
            act_type = ""
        elif ann.get("result_action_type", "") == 2:
            act_type = ""
        elif ann.get("result_action_type", "") == 3:
            act_type = "type"
        elif ann.get("result_action_type", "") == 4:
            act_type = "scroll"
        elif ann.get("result_action_type", "") == 5:
            act_type = "press back"
        elif ann.get("result_action_type", "") == 6:
            act_type = "press home"
        elif ann.get("result_action_type", "") == 7:
            act_type = "press enter"
        elif ann.get("result_action_type", "") == 10:
            act_type = "completed"

        instruction=ann.get("instruction", "")
        screen_desc=ann.get("coat_screen_desc", "")
        correct_next_action = (
            f"result_action_type: {act_type}\n"
            f"result_action_text: {ann.get('result_action_text','')}\n"
            f"result_touch_yx: {ann.get('result_touch_yx','')}\n"
            f"result_lift_yx: {ann.get('result_lift_yx','')}")

        return img_path, instruction, screen_desc, correct_next_action

    def _prepare_model(self):
        """
        Load pretrained model and set up LoRA fine-tuning.
        """
        print(f"Loading pretrained model from {self.pretrained_model_path}")
        self.processor = VLChatProcessor.from_pretrained(
            self.pretrained_model_path,
            slow_image_processor_class="AutoImageProcessor"
        )
        self.model = EnhancedMultiModalModel.from_pretrained(
            self.pretrained_model_path,
            device_map="auto"
        )

        # LoRA
        lora_config = LoraConfig(**self.lora_config)
        self.model = get_peft_model(self.model, lora_config)
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable params: {trainable_params} / {total_params}")

    def _prepare_optimizer_and_scheduler(self, dataset_size: int):
        """"
        Prepare optimizer and learning rate scheduler.
        """
        if self.optimizer_name == "AdamW":
            optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.0, betas=(0.9, 0.95))
        elif self.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

        num_training_steps = dataset_size * self.max_epochs // self.batch_size
        num_warmup_steps = int(0.1 * num_training_steps)

        scheduler_name = self.training_args.get("lr_scheduler_type", "cosine")
        if scheduler_name not in ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]:
            raise ValueError(f"Unsupported learning rate scheduler: {scheduler_name}")

        lr_scheduler = get_scheduler(
            name=scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Custom data collate function.
        """
        conversations = []
        for item in batch:
            conversations.extend(self._generate_conversation(item["image_path"], item["instruction"], item["screen_description"], item["action"]))
        
        pil_images_list = load_pil_images(conversations)
        encoded = self.processor(
            conversations=conversations,
            images=pil_images_list,
            return_tensors="pt",
            force_batchify=True,
        )
        encoded["labels"] = encoded["input_ids"].clone()

        image_placeholder_token_id = self.processor.tokenizer.convert_tokens_to_ids("<image_placeholder>")
        input_ids = encoded["input_ids"]
        image_token_masks = (input_ids == image_placeholder_token_id)

        if not image_token_masks.any():
            raise ValueError("No <image_placeholder> tokens found in the input!")

        encoded["image_token_masks"] = image_token_masks
        return dict(encoded)

    def _generate_conversation(self, image_path: str, instruction: str, screen_description: str, action: str) -> List[Dict[str, Any]]:
        """
        Generate conversation template for each sample.
        """
        description_conversation = [
            {"role": "<|User|>", "content": f"<image_placeholder>\n{system_desc_message.format(instruction=instruction)}", "images": [image_path]},
            {"role": "<|Assistant|>", "content": screen_description},
        ]
        action_conversation =  [
            {"role": "<|User|>", "content": f"<image_placeholder>\n{system_act_message.format(instruction=instruction, screen_desc=screen_description)}", "images": [image_path]},
            {"role": "<|Assistant|>", "content": action},
        ]
        # print("conversation : ", conversation)
        return action_conversation

    def train(self):
        """
        Main training flow.
        """
        pairs = self._load_data()
        dataset = [{"image_path": img, "instruction": inst, "screen_description": desc, "action": act} for img, inst, desc, act in pairs]

        self._prepare_model()
        self._prepare_optimizer_and_scheduler(len(dataset))

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.max_epochs,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.lr,
            **self.training_args,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=self._collate_fn,
            optimizers=(self.optimizer, self.lr_scheduler),
        )

        print("[Start!] Start training!!!!!---------->>>>>>>")
        trainer.train()

        print("[on Progress] Merging LoRA weights...")
        self.model = self.model.merge_and_unload()

        print("[on Progress] Saving...")
        self.model.save_pretrained(self.output_dir)
        self.processor.save_pretrained(self.output_dir)
        print(f"[Done!] Fine-tuned model have saved to {self.output_dir}")
