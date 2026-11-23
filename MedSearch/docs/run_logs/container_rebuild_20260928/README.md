# llm-pytorch 容器恢复记录（2026-09-28）

旧容器 `c8f4518d842b` 的 Python 包清单见 `pip_packages.json`（329 项），APT 手动安装包清单见 `apt_manual.txt`。清理前，旧容器还含有 `/tmp/medsearch_eval`（约 47 GB）、`/tmp/medsearch_teacher_7b`（约 33 GB）、`/tmp/medsearch_student_3b`（约 6.3 GB）等副本。两个模型目录的全部模型分片与主机原件逐个 SHA256 一致；这些临时副本已删除。

旧容器软件环境保存为本地镜像 `medsearch/llm-pytorch-env:20260928`（ID `sha256:aae04d4b36144f7db348cfa83e7df80e8c0d3d8da1e88abef4663aca5462d030`）。新容器 ID 为 `02f830472e38`，名称仍为 `llm-pytorch`。旧容器已删除；两个命名卷 `llm_huggingface_cache`、`llm_torch_cache` 保留，`cotar-autoresearch` 未动。

新容器使用 GPU、16 GiB 共享内存，并把主机 `D:\code_list` 绑定到 `/workspace/code_list`。创建命令（CMD）：

```cmd
docker run -dit --name llm-pytorch --gpus all --shm-size 16g --mount type=bind,source=D:\code_list,target=/workspace/code_list --mount type=volume,source=llm_huggingface_cache,target=/root/.cache/huggingface --mount type=volume,source=llm_torch_cache,target=/root/.cache/torch --workdir /workspace/code_list medsearch/llm-pytorch-env:20260928 bash
```

新容器创建后已验证：能读取 `/workspace/code_list/some tricks/LLMLeanring/AGENTS.md`；识别 RTX 5090（32607 MiB）；新旧容器的 329 个 Python 包名称和版本逐项一致。

Docker 数据 VHDX：`D:\code_list\docker-data\DockerDesktopWSL\disk\docker_data.vhdx`。删除容器内文件、旧容器并运行 `fstrim` 后，Windows D 盘空闲空间没有立即增加。2026-09-28 使用 DiskPart 的 `attach vdisk readonly` → `compact vdisk` → `detach vdisk`，D 盘空闲从 17,676,083,200 字节增加到 110,834,720,768 字节，实测回收 93,158,637,568 字节。此前从约 6.27 GB 到 17.68 GB 的增长来自用户清理其他目录，**不计入 Docker 回收**。

后续实验必须先检查主机 D 盘空闲空间和临时空间预算；模型、索引只使用主机现有文件的挂载路径。挂载不可读时先修复挂载，不向容器可写层复制大型模型。评测结果、配置与日志写入主机挂载目录。
