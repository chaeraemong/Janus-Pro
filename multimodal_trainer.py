# multimodal_trainer.py

import os, json
from typing import List, Tuple, Dict, Any
import torch
from glob import glob
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW, SGD
from transformers import Trainer, TrainingArguments, get_scheduler
from models import MultiModalityCausalLM, VLChatProcessor
from utils.io import load_pil_images


system_message = """You are a GUI agent.
I will give you a screenshot of a mobile phone.
**INSTRUCTION**: {instruction}
**SCREEN DESCRIPTION**: {screen_desc}

**SCREEN COORDINATE SYSTEM**: A coordinate (x, y) represents a point on the screen. The first value, labeled as `x`, horizontal,i.e. x ranges from 0 to 1, meaning the position of point ranges from the left to right, where x<0.4 means left, 0.4<=x<=0.6 means middle and x>0.6 meansright. The second value, labeled as `y`, is vertical, i.e. y ranges from 0 to 1, meaning the position of point ranges from the bottom to top. where y<0.2means bottom, 0.2<=y<0.4 means lower, + 0.4<=y<0.5 means lower middle, 0.5<=y<=0.6 means upper middle, 0.6<y<=0.8 means upper, and y>0.8 means top. 
**TASK**: Given the screenshot and instruction, follow the bellow tasks to fulfill the instruction.
1. You should analyze the screen for relevant details that might pertain to the given query. This includes checking for specific applications, icons, or buttons that are visible, and any information or results that are currently displayed on the screen. The screen analysis should be like the **SCREEN DESCRIPTION** part.
2. After screen description through screen analysis, describe possible actions you may conduct. You must answer by two sentences with the format: 'Think: ... Possible actions are ...'.
3. Based on the possible actions conductable, you have to perform a final action on screen.
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
        이미지와 텍스트의 결합 처리를 지원.
        """
        # 이미지 입력과 마스크가 제공되면 멀티모달 입력을 처리
        if pixel_values is not None and image_token_masks is not None:
            inputs_embeds = self._process_multimodal_inputs(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_token_masks=image_token_masks,
            )
        else:
            # 텍스트 임베딩만 사용하는 경우
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 이미 생성된 inputs_embeds를 사용
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
        이미지 토큰과 텍스트 토큰을 결합한 임베딩을 생성.

        Args:
            input_ids (torch.LongTensor): [batch_size, seq_length]
            pixel_values (torch.FloatTensor): [batch_size, num_images, 3, height, width]
            image_token_masks (torch.BoolTensor): [batch_size, seq_length]

        Returns:
            torch.Tensor: 결합된 토큰 임베딩 [batch_size, seq_length, hidden_dim]
        """
        bs, n = pixel_values.shape[0:2]
        images = pixel_values.view(bs * n, *pixel_values.shape[2:])
        image_features = self.vision_model(images)                  # 비전 모델을 통해 이미지 특징 추출
        aligned_features = self.aligner(image_features)             # alignment model 적용

        aligned_features = aligned_features.view(bs, n, *aligned_features.shape[1:])    # [batch_size, num_images, token_length, hidden]
        aligned_features = aligned_features.flatten(1, 2)           # 이미지 토큰 차원 합치기

        text_embeds = self.language_model.get_input_embeddings()(input_ids)             # 텍스트 임베딩 생성

        for i in range(bs):
            num_image_tokens = image_token_masks[i].sum().item()    # 마스크된 위치에 이미지 임베딩 삽입
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
                 batch_size: int = 1, 
                 max_epochs: int = 10, 
                 lr: float = 3e-4, 
                 user_question: str = system_message,
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
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
        self.training_args = training_args if training_args else {
            "fp16": True,
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
        # 1) 에피소드 폴더 목록을 가져오고
        episode_dirs = [
            os.path.join(self.data_dir, d)
            for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d))
        ]
        # 2) 각 폴더 안의 모든 .json 파일을 읽어서
        for ep in episode_dirs:
            for jp in glob(os.path.join(ep, "*.json")):
                with open(jp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # JSON이 리스트라면 각 원소, 아니면 하나의 dict
                entries = data if isinstance(data, list) else [data]
                for ann in entries:
                    # ann["image_path"]는 "general/...png" 형태라고 가정
                    img_rel = ann["image_path"]
                    prefix  = "general" + os.sep
                    if img_rel.startswith(prefix):
                        img_rel = img_rel[len(prefix):]
                    img_path = os.path.join(self.data_dir, img_rel)
                    self.samples.append((img_path, ann))
        pairs = []
        for i in range(self.__len__()):
            image_path, inst, screen_desc, action = self.__getitem__(i)
            pairs.append((image_path, inst, screen_desc, action))

        return pairs
    
    def __len__(self):
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

        instruction=ann.get("instruction", ""),
        screen_desc=ann.get("coat_screen_desc", ""),
        correct_next_action = (
            f"result_action_type: {act_type}\n"
            f"result_action_text: {ann.get('result_action_text','')}\n"
            f"result_touch_yx: {ann.get('result_touch_yx','')}\n"
            f"result_lift_yx: {ann.get('result_lift_yx','')}")

        return img_path, instruction, screen_desc, correct_next_action

    def _prepare_model(self):
        """
        사전학습 모델을 로드하고 LoRA 파인튜닝 설정을 수행.
        """
        print(f"Loading pretrained model from {self.pretrained_model_path}")
        self.processor = VLChatProcessor.from_pretrained(
            self.pretrained_model_path,
            slow_image_processor_class="AutoImageProcessor"
        )
        self.model = EnhancedMultiModalModel.from_pretrained(
            self.pretrained_model_path,
            #torch_dtype=torch.float16,
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
        옵티마이저 및 학습률 스케줄러를 준비.
        """
        if self.optimizer_name == "AdamW":
            optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
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
        사용자 정의 데이터 콜레이트 함수.
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
        대화 템플릿을 생성.
        """
        return [
            {"role": "<|User|>", "content": f"<image_placeholder>\n{system_message.format(instruction=instruction, screen_desc=screen_description)}", "images": [image_path]},
            {"role": "<|Assistant|>", "content": action},
        ]

    def train(self):
        """
        메인 학습 흐름.
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

        print("[on Progress] Saving...")
        new_model_dir = "./janus_lora/ver."
        self.model.save_pretrained(self.output_dir)
        self.processor.save_pretrained(self.output_dir)
        print(f"[Done!] Fine-tuned model have saved to {self.output_dir}")
