from utils.multimodal_trainer_fp16 import EnhancedMultiModalTrainer

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

trainer = EnhancedMultiModalTrainer(
    data_dir="./data/android_in_the_zoo/train/general",
    pretrained_model_path="deepseek-ai/Janus-Pro-1B",
    output_dir="./janus_lora",
    batch_size=2,
    max_epochs=10,
    lr=3e-4,
    user_question=system_message,
    optimizer_name="AdamW",
    lora_config={
        "r": 16,
        "lora_alpha": 32,
        "target_modules": [
            # vision_model
            "vision_model.vision_tower.patch_embed.proj",
            "vision_model.vision_tower.blocks.*.attn.qkv",
            "vision_model.vision_tower.blocks.*.attn.proj",
            "vision_model.vision_tower.blocks.*.mlp.fc1",
            "vision_model.vision_tower.blocks.*.mlp.fc2",
            "vision_model.vision_tower.attn_pool.q",
            "vision_model.vision_tower.attn_pool.kv",
            "vision_model.vision_tower.attn_pool.proj",
            "vision_model.vision_tower.attn_pool.mlp.fc1",
            "vision_model.vision_tower.attn_pool.mlp.fc2",

            # aligner
            "aligner.layers.0",
            "aligner.layers.2",

            # language_model
            "language_model.model.embed_tokens",
            "language_model.model.layers.*.self_attn.q_proj",
            "language_model.model.layers.*.self_attn.k_proj",
            "language_model.model.layers.*.self_attn.v_proj",
            "language_model.model.layers.*.self_attn.o_proj",
            "vl_gpt.language_model.model.layers.*.mlp.gate_proj",
            "vl_gpt.language_model.model.layers.*.mlp.up_proj",
            "vl_gpt.language_model.model.layers.*.mlp.down_proj",
            "language_model.lm_head",
        ],
        "lora_dropout": 0.1,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },
    training_args={
        "fp16": True,
        "max_grad_norm": 1.0,
        "save_strategy": "epoch",
        "evaluation_strategy": "no",
        "logging_steps": 50,
        "save_total_limit": 2,
        "remove_unused_columns": False,
    }
)
trainer.train()
