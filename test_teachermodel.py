import os
import sys
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import deepspeed
import datasets
from tqdm import tqdm

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
    parser.add_argument('--data_path', type=str, nargs='+', required=True, help='Path to preprocessed data')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for testing')
    parser.add_argument('--max_seq_length', type=int, default=2048, help='Maximum sequence length')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to test')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--deepspeed', action='store_true', help='Use DeepSpeed')
    parser.add_argument('--deepspeed_stage', type=int, default=2, help='DeepSpeed ZeRO stage')
    parser.add_argument('--offload', action='store_true', help='Use CPU offloading')
    parser.add_argument('--fp32', action='store_true', help='Use FP32 instead of BF16')
    parser.add_argument('--need_to_pad',action='store_true',default=False,help='whether to pad the input with other sample to fill the sample to max length')
    return parser.parse_args()

def data_collator_with_pad(examples, max_seq_length, pad_token_id):
    input_ids = [example['input_ids'] for example in examples]
    attention_mask = [example['attention_mask'] for example in examples]
    
    # 计算最大长度，但不超过max_seq_length
    max_len = min(max([len(ids) for ids in input_ids]), max_seq_length)
    
    # 填充到相同长度
    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []
    
    for ids, mask in zip(input_ids, attention_mask):
        # 截断到最大长度
        if len(ids) > max_len:
            ids = ids[:max_len]
            mask = mask[:max_len]
        
        # 填充
        padding_length = max_len - len(ids)
        padded_ids = ids + [pad_token_id] * padding_length
        padded_mask = mask + [0] * padding_length
        
        padded_input_ids.append(padded_ids)
        padded_attention_mask.append(padded_mask)
        padded_labels.append(padded_ids)  # 使用input_ids作为labels
    
    return {
        'input_ids': torch.tensor(padded_input_ids),
        'attention_mask': torch.tensor(padded_attention_mask),
        'labels': torch.tensor(padded_labels)
    }

def setup_distributed(args):
    """设置分布式环境"""
    # 如果local_rank不是通过环境变量设置的，则使用args中的值
    if 'LOCAL_RANK' not in os.environ and args.local_rank != -1:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    
    # 如果是通过DeepSpeed启动的，则local_rank已经在环境变量中
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        args.local_rank = 0  # 默认为主进程
    
    # 设置WORLD_SIZE
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = str(args.num_gpus)
    
    # 设置RANK
    if 'RANK' not in os.environ:
        os.environ['RANK'] = str(args.local_rank)
    
    print(f"Distributed setup: local_rank={args.local_rank}, world_size={os.environ['WORLD_SIZE']}")
    
    # 初始化分布式环境
    if args.deepspeed:
        try:
            deepspeed.init_distributed()
        except ModuleNotFoundError as e:
            if 'mpi4py' in str(e):
                print("mpi4py not found, initializing without MPI support")
                if not torch.distributed.is_initialized():
                    torch.distributed.init_process_group(backend='nccl', init_method='env://')
            else:
                raise e
    elif args.num_gpus > 1:
        # 如果不使用DeepSpeed但需要多GPU，则使用PyTorch的分布式
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
    
    # 设置当前设备
    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)

def test_teacher_model(args):
    # 设置设备和数据类型
    dtype = torch.float32 if args.fp32 else torch.bfloat16
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    
    # 只在主进程上打印信息
    is_main_process = args.local_rank == 0
    
    if is_main_process:
        print(f"Loading tokenizer from {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if is_main_process:
        print(f"Loading model from {args.model_id}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=dtype,
            device_map='cpu',  # 先加载到CPU
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # 检查模型是否正确加载
        if is_main_process:
            print(f"Model loaded successfully, type: {type(model)}")
            # 检查一些关键参数是否正常
            for name, param in list(model.named_parameters())[:5]:
                print(f"Parameter {name}: shape={param.shape}, mean={param.mean().item():.6f}, std={param.std().item():.6f}")
        
        model.eval()
        for name, param in model.named_parameters():
            param.requires_grad = False
            
    except Exception as e:
        if is_main_process:
            print(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
        return
    
    # 使用DeepSpeed包装模型（如果需要）
    if args.deepspeed:
        ds_config = {
            "distributed_backend": "nccl",
            "train_batch_size": args.batch_size * args.num_gpus,  # 全局批量大小
            "bf16": {
                "enabled": not args.fp32
            },
            "fp32": {
                "enabled": args.fp32
            },
            "zero_optimization": {
                "stage": args.deepspeed_stage,
                "stage3_max_live_parameters": 1e10,
                "stage3_max_reuse_distance": 1e10,
                "stage3_prefetch_bucket_size": "auto",
                "memory_efficient_linear": True,
                "stage3_param_persistence_threshold": "auto",
                # "zero_hpz_partition_size": args.num_gpus,
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True,
                    "buffer_count": 5,
                    "buffer_size": 1e9
                } if args.offload else None,
                # "allgather_partitions": True,
                "sub_group_size": 1e8,
                # "reduce_scatter": True,
                "reduce_bucket_size": 5e6,
                "overlap_comm": True,
                "contiguous_gradients": True
            },
            "zero_force_ds_cpu_initialization": True,
            "zero_allow_untested_optimizer": True,
            "wall_clock_breakdown": False,
            "dump_state": True
        }
        
        model_engine, _, _, _ = deepspeed.initialize(
            model=model,
            config=ds_config
        )
        model = model_engine
    else:
        model = model.to(device)
    
    # 加载数据集
    if is_main_process:
        print(f"Loading dataset from {args.data_path}")
    try:
        from data.raw_dataset import load_datasets_from_directories,TypedDataset,TypedStreamingCLMDataCollator
        all_ds,feature_types = load_datasets_from_directories(args.data_path,tokenizer)
        typed_dataset = TypedDataset(all_ds, feature_types)
        # print(all_ds)
        # con_ds = datasets.concatenate_datasets(all_ds)
        # data_collator = StreamingCLMDataCollator(tokenizer=tokenizer, max_length=args.max_seq_length)
        data_collator = TypedStreamingCLMDataCollator(tokenizer=tokenizer, 
                                                  max_length=args.max_seq_length, 
                                                  min_length=args.max_seq_length, 
                                                  typed_dataset=typed_dataset,
                                                  need_to_pad=args.need_to_pad)
        from torch.utils.data.distributed import DistributedSampler
        if is_main_process:
            print(f"Dataset loaded successfully with {len(typed_dataset)} samples")
            # print(f"Dataset features: {typed_dataset.features}")
        
        # 创建数据加载器
        from torch.utils.data.dataloader import DataLoader
        from torch.utils.data.distributed import DistributedSampler
        from functools import partial
        
        # 如果是分布式训练，使用DistributedSampler
        if args.num_gpus > 1:
            sampler = DistributedSampler(
                typed_dataset,
                num_replicas=args.num_gpus,
                rank=args.local_rank,
                shuffle=True
            )
            shuffle = False  # 使用sampler时不能设置shuffle=True
        else:
            sampler = None
            shuffle = True
        
        # collate_fn = partial(data_collator_with_pad, max_seq_length=args.max_seq_length, pad_token_id=tokenizer.pad_token_id)
        dataloader = DataLoader(
            typed_dataset, 
            batch_size=args.batch_size, 
            sampler=sampler,
            collate_fn=data_collator,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True
        )
        
    except Exception as e:
        if is_main_process:
            print(f"Failed to load dataset: {e}")
            import traceback
            traceback.print_exc()
        return
    
    # 测试模型前向传播
    if is_main_process:
        print("Testing model forward pass...")

    
    # 然后在实际数据上测试
    if is_main_process:
        print("\n=== Testing with actual dataset ===")
    success_count = 0
    nan_count = 0
    error_count = 0
    
    # 使用tqdm只在主进程显示进度条
    dataloader_iter = tqdm(dataloader, desc="Testing batches") if is_main_process else dataloader
    
    for i, batch in enumerate(dataloader_iter):
        if i >= args.num_samples:
            break
        
        try:
            # 将数据移动到正确的设备
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            # 打印输入形状（只在主进程）
            if is_main_process:
                print(f"\nBatch {i+1}:")
                print(f"input_ids shape: {batch['input_ids'].shape}")
                print(f"attention_mask shape: {batch['attention_mask'].shape}")
                print(f"labels shape: {batch['labels'].shape}")
                
                # 检查输入是否包含NaN
                if torch.isnan(batch['input_ids']).any():
                    print("WARNING: input_ids contains NaN values")
                if torch.isnan(batch['attention_mask']).any():
                    print("WARNING: attention_mask contains NaN values")
                if torch.isnan(batch['labels']).any():
                    print("WARNING: labels contains NaN values")
            
            # 前向传播
            with torch.no_grad():
                outputs = model(
                    input_ids=batch['input_ids'],
                    # attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                    use_cache=False
                )
                
                # 检查输出（只在主进程）
                if is_main_process:
                    if torch.isnan(outputs.logits).any():
                        print("WARNING: NaN values detected in logits")
                        nan_count += 1
                        
                        # 尝试查找NaN的位置
                        nan_indices = torch.isnan(outputs.logits).nonzero()
                        if len(nan_indices) > 0:
                            print(f"NaN positions (sample): {nan_indices[:5]}")
                            
                        # 检查输入中对应位置的值
                        for idx in nan_indices[:5]:
                            batch_idx, seq_idx, vocab_idx = idx
                            print(f"Input at position {batch_idx}, {seq_idx}: {batch['input_ids'][batch_idx, seq_idx].item()}")
                            
                    if torch.isnan(outputs.loss).any():
                        print("WARNING: NaN value detected in loss")
                        nan_count += 1
                    else:
                        print(f"Loss: {outputs.loss.item()}")
                        print(f"Logits shape: {outputs.logits.shape}")
                        print(f"Logits stats: min={outputs.logits.min().item()}, max={outputs.logits.max().item()}, mean={outputs.logits.mean().item()}")
                        success_count += 1
                    
        except Exception as e:
            if is_main_process:
                print(f"Error processing batch {i+1}: {e}")
                import traceback
                traceback.print_exc()
            error_count += 1
    
    # 打印总结（只在主进程）
    if is_main_process:
        print("\n=== Test Summary ===")
        print(f"Total batches tested: {min(args.num_samples, len(dataloader))}")
        print(f"Successful batches: {success_count}")
        print(f"Batches with NaN: {nan_count}")
        print(f"Batches with errors: {error_count}")

if __name__ == "__main__":
    setup_env()
    args = parse_args()
    
    # 设置分布式环境
    setup_distributed(args)
    
    test_teacher_model(args) 