import os
import sys
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
# import datasets
# from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.strategies import DeepSpeedStrategy,FSDPStrategy
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
from llama_cpp import Llama

# 设置分布式环境变量
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

def setup_env():
    parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    rwkv_insidea_path = os.path.join(parent_dir, 'rwkv_inside')
    sys.path.append(rwkv_insidea_path)
    sys.path.append(parent_dir)
    print(f'add path: {rwkv_insidea_path} to sys.path')
    # os.environ['NCCL_DEBUG'] = 'INFO'
    # os.environ['NCCL_BLOCKING_WAIT'] = '1'
    # os.environ['NCCL_P2P_LEVEL'] = 'NVL'
    # os.environ['NCCL_NVLS_DISABLE'] = '1'
    # os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'


def parse_args():
    parser = argparse.ArgumentParser(description='Test teacher model forward pass')
    parser.add_argument('--model_id', type=str, required=True, help='Teacher model ID or path')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Tokenizer path')
    parser.add_argument('--data_path', type=str, nargs='+', required=True, help='Path to preprocessed data')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for testing')
    parser.add_argument('--max_seq_length', type=int, default=2048, help='Maximum sequence length')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to test')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--fp32', action='store_true', help='Use FP32 instead of BF16')
    parser.add_argument('--need_to_pad',action='store_true',default=False,help='whether to pad the input with other sample to fill the sample to max length')
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes for distributed training')
    return parser.parse_args()


class TeacherModelTester(pl.LightningModule):
    def __init__(self, model, tokenizer, args):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.success_count = 0
        self.nan_count = 0
        self.error_count = 0
        self.batch_count = 0
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False
        )
    
    def test_step(self, batch, batch_idx):
        if self.batch_count >= self.args.num_samples:
            return None
        
        self.batch_count += 1
        
        try:
            # 前向传播
            outputs = self(
                input_ids=batch['input_ids'],
                # attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            # 检查输出
            has_nan = False
            
            if torch.isnan(outputs.logits).any():
                self.print("WARNING: NaN values detected in logits")
                has_nan = True
                
                # 尝试查找NaN的位置
                nan_indices = torch.isnan(outputs.logits).nonzero()
                if len(nan_indices) > 0:
                    self.print(f"NaN positions (sample): {nan_indices[:5]}")
                    
                # 检查输入中对应位置的值
                for idx in nan_indices[:5]:
                    batch_idx, seq_idx, vocab_idx = idx
                    self.print(f"Input at position {batch_idx}, {seq_idx}: {batch['input_ids'][batch_idx, seq_idx].item()}")
                    
            if torch.isnan(outputs.loss).any():
                self.print("WARNING: NaN value detected in loss")
                has_nan = True
            
            if has_nan:
                self.nan_count += 1
            else:
                self.print(f"Loss: {outputs.loss.item()}")
                self.print(f"Logits shape: {outputs.logits.shape}")
                self.print(f"Logits stats: min={outputs.logits.min().item()}, max={outputs.logits.max().item()}, mean={outputs.logits.mean().item()}")
                self.success_count += 1
                
            return {"loss": outputs.loss}
                
        except Exception as e:
            self.print(f"Error processing batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            self.error_count += 1
            return None
    
    def on_test_epoch_end(self):
        self.print("\n=== Test Summary ===")
        self.print(f"Total batches tested: {self.batch_count}")
        self.print(f"Successful batches: {self.success_count}")
        self.print(f"Batches with NaN: {self.nan_count}")
        self.print(f"Batches with errors: {self.error_count}")
    
    def configure_optimizers(self):
        # 由于只是前向传播，不需要优化器
        return None


def test_teacher_model(args):
    # 设置设备和数据类型
    dtype = torch.float32 if args.fp32 else torch.bfloat16
    
    # 加载tokenizer
    print(f"Loading tokenizer from {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading model from {args.model_id}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=dtype,
            device_map='cpu',  # 先加载到CPU
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        # model = Llama(
        #     model_path=args.model_id,
        # )
        
        # 检查模型是否正确加载
        print(f"Model loaded successfully, type: {type(model)}")
        # 检查一些关键参数是否正常
        for name, param in list(model.named_parameters())[:5]:
            print(f"Parameter {name}: shape={param.shape}, mean={param.mean().item():.6f}, std={param.std().item():.6f}")
        
        model.eval()
        for name, param in model.named_parameters():
            param.requires_grad = False
            
    except Exception as e:
        print(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 加载数据集
    print(f"Loading dataset from {args.data_path}")
    try:
        from data.raw_dataset import load_datasets_from_directories, TypedDataset, TypedStreamingCLMDataCollator
        all_ds, feature_types = load_datasets_from_directories(args.data_path, tokenizer)
        typed_dataset = TypedDataset(all_ds, feature_types)
        
        data_collator = TypedStreamingCLMDataCollator(
            tokenizer=tokenizer, 
            max_length=args.max_seq_length, 
            min_length=args.max_seq_length, 
            typed_dataset=typed_dataset,
            need_to_pad=args.need_to_pad
        )
        
        print(f"Dataset loaded successfully with {len(typed_dataset)} samples")
        
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 创建PyTorch Lightning模型
    teacher_model = TeacherModelTester(model, tokenizer, args)
    
    # 设置数据加载器
    def collate_fn(batch):
        return data_collator(batch)
    
    # 创建PyTorch Lightning的DataLoader
    dataloader = DataLoader(
        typed_dataset, 
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # 创建DeepSpeed配置
    # ds_config = {
    #     "zero_optimization": {
    #         "stage": 3,  # 这里指定Zero stage (1,2,3)
    #         "offload_optimizer": {
    #             "device": "cpu",
    #             "pin_memory": True
    #         },
    #         "offload_param": {
    #             "device": "cpu",
    #             "pin_memory": True
    #         },
    #         "overlap_comm": True,
    #         "contiguous_gradients": True,
    #         "sub_group_size": 1e9,
    #         "reduce_bucket_size": "auto",
    #         "stage3_prefetch_bucket_size": "auto",
    #         "stage3_param_persistence_threshold": "auto",
    #         "stage3_max_live_parameters": 1e9,
    #         "stage3_max_reuse_distance": 1e9,
    #         "gather_16bit_weights_on_model_save": True
    #     },
    #     "bf16": {
    #         "enabled": not args.fp32
    #     },
    #     "fp16": {
    #         "enabled": False
    #     },
    #     "train_micro_batch_size_per_gpu": args.batch_size,
    # }
    ds_config = {
        "distributed_backend": "nccl",
        "train_batch_size": args.batch_size,
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 2,
            # "stage3_max_live_parameters": 1e9,
            # "stage3_max_reuse_distance": 1e9,
            # "stage3_prefetch_bucket_size": 5e6,
            "memory_efficient_linear": True,
            # "stage3_param_persistence_threshold": 1e5,
            # "offload_param": {
            #     "device": "cpu",
            #     "pin_memory": True,
            #     "buffer_count": 4,
            #     "buffer_size": 1e8
            # },
            "allgather_partitions": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e6,
            "overlap_comm": True,
            "contiguous_gradients": True
        },
        "zero_force_ds_cpu_initialization": True,
        "zero_allow_untested_optimizer": True,
        "dump_state": True
    }
    # ds_config = {
    #     "train_batch_size": args.batch_size*args.num_gpus,
    #     "bf16": {
    #         "enabled": not args.fp32
    #     },
    #     "fp16": {
    #         "enabled": False
    #     },
    #     "zero_optimization": {
    #         "stage": 3
    #     },
    #     "zero_allow_untested_optimizer": True,
    #     "steps_per_print": 10
    # }
    #创建DeepSpeed策略
    strategy = DeepSpeedStrategy(config=ds_config)
    # if not args.fp32:
    #     mixed_precision_policy = MixedPrecision(
    #         param_dtype=torch.bfloat16,
    #         reduce_dtype=torch.bfloat16,
    #         buffer_dtype=torch.bfloat16,
    #     )
    # else:
    #     mixed_precision_policy = None
    # strategy = FSDPStrategy(
    #     auto_wrap_policy=None,
    #     activation_checkpointing=None,
    #     cpu_offload=True,
    #     sharding_strategy="HYBRID_SHARD",
    #     device_mesh=(1, args.num_gpus),
    #     mixed_precision=mixed_precision_policy  # 使用正确的混合精度配置
    # )

    # 创建Trainer
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.num_gpus if torch.cuda.is_available() else None,
        num_nodes=args.num_nodes,
        precision="32" if args.fp32 else "bf16",
        strategy=strategy,
        logger=False,
        enable_progress_bar=True,
        enable_model_summary=True,
        max_epochs=1,  # 只运行一个epoch
    )
    
    # 运行测试
    print("Testing model forward pass...")
    trainer.test(teacher_model, dataloader)


if __name__ == "__main__":
    setup_env()
    args = parse_args()
    test_teacher_model(args)