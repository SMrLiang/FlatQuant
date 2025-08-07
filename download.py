import os
import requests
import tqdm
# 设置目标路径
save_dir = "/data/yihengliang/q_data/wikitext"  # 替换成你想保存的路径
os.makedirs(save_dir, exist_ok=True)

# 文件 URL（来自 C4 的 en 分区）

# url = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/train-00000-of-00001.parquet"
url = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet"
target_file = os.path.join(save_dir, "test-00000-of-00001.parquet")

# 下载文件
with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open(target_file, "wb") as f:
        for chunk in tqdm.tqdm(r.iter_content(chunk_size=8192)):
            f.write(chunk)

print(f"Downloaded to {target_file}")

