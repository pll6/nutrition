import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from mmseg.models.builder import LOSSES

def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features

@LOSSES.register_module()
class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2
        return total_loss

@LOSSES.register_module()
class SingleGPUCrossBatchLoss(nn.Module):
    def __init__(self, feature_dim=512, queue_size=1024):
        """
        queue_size: 负样本队列的大小。设为 1024 相当于你拥有了 Global Batch Size = 1024 的负样本池！
        """
        super().__init__()
        self.queue_size = queue_size
        
        # 注册一个不参与梯度更新的 Buffer 作为队列，用随机噪声初始化并归一化
        self.register_buffer("text_queue", torch.randn(queue_size, feature_dim))
        self.text_queue = F.normalize(self.text_queue, dim=-1)
        
        # 队列指针
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """更新历史文本队列：把当前的挤进去，最老的挤出去"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # 防止越界的小处理
        if ptr + batch_size <= self.queue_size:
            self.text_queue[ptr:ptr + batch_size] = keys
            self.queue_ptr[0] = (ptr + batch_size) % self.queue_size
        else:
            remain = self.queue_size - ptr
            self.text_queue[ptr:] = keys[:remain]
            self.text_queue[:batch_size - remain] = keys[remain:]
            self.queue_ptr[0] = batch_size - remain

    def forward(self, image_features, text_features, logit_scale):
        # image_features: [B, 512]
        # text_features:  [B, 512]
        B = image_features.shape[0]
        device = image_features.device

        # 1. 魔法发生地：把当前的文本特征，和队列里的上千个历史文本特征拼接起来！
        # 维度变成 [B + queue_size, 512]
        all_text_features = torch.cat([text_features, self.text_queue.clone().detach()], dim=0)

        # 2. 算相似度：[B, 512] @ [512, B + queue_size] -> [B, B + queue_size]
        # 现在，你的每一张图片，都在和成百上千个文本算相似度了！
        logits_per_image = logit_scale * image_features @ all_text_features.T

        # 3. 找正确答案：对于图像来说，正样本永远是当前 Batch 里对角线上的那个文本 (索引 0 到 B-1)
        labels = torch.arange(B, device=device, dtype=torch.long)

        # 4. 算 Loss (单卡加了 Queue 之后，通常只算单向的 Image->Text 分类 Loss 就足够强大了)
        loss = F.cross_entropy(logits_per_image, labels)

        # 5. 把当前 Batch 的文本特征无梯度地塞进队列，供下个 Batch 使用
        self._dequeue_and_enqueue(text_features)

        return loss