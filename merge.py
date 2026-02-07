import torch
import os
from torch.distributed.checkpoint import FileSystemReader, load_state_dict

def merge_fsdp_shards(checkpoint_dir, output_path):
    print(f"Reading from {checkpoint_dir} ...")
    
    reader = FileSystemReader(checkpoint_dir)
    metadata = reader.read_metadata()
    
    full_state_dict = {}
    
    for key, item in metadata.state_dict_metadata.items():
        if hasattr(item, 'size'):
            full_state_dict[key] = torch.empty(item.size, dtype=item.properties.dtype)
        elif hasattr(item, 'shape'):
            full_state_dict[key] = torch.empty(item.shape, dtype=item.type)
        else:
            full_state_dict[key] = None

    load_state_dict(
        state_dict=full_state_dict,
        storage_reader=reader,
        no_dist=True,
    )

    cleaned_state_dict = {}
    for k, v in full_state_dict.items():
        if k.startswith("model."):
            new_key = k[6:] 
            cleaned_state_dict[new_key] = v
        elif k.startswith("optimizer."):
            continue
        else:
            cleaned_state_dict[k] = v

    save_obj = {"model": cleaned_state_dict}
    
    torch.save(save_obj, output_path)
    print(f"✅ Succeed! File saved to: {output_path}")

if __name__ == "__main__":
    shard_dir = "latest.pth"
    out_file = "unified_model.pth"
    
    if os.path.exists(shard_dir):
        try:
            merge_fsdp_shards(shard_dir, out_file)
        except Exception as e:
            import traceback
            traceback.print_exc()
    else:
        print(f"错误: 找不到目录 {shard_dir}")
