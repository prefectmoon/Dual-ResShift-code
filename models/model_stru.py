'''

C:\Users\admin\anaconda3\envs\zz\python.exe E:\code\ResShift-unet\models\unetv3.py
SuperResModel image_size： 64
SuperResModel(
  (time_embed): Sequential(
    (0): Linear(in_features=96, out_features=384, bias=True)
    (1): SiLU()
    (2): Linear(in_features=384, out_features=384, bias=True)
  )
  (input_blocks): ModuleList(
    (0): TimestepEmbedSequential(
      (0): Conv2d(3, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (1-2): 2 x TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 96, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Identity()
        (x_upd): Identity()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=192, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 96, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Identity()
      )
    )
    (3): TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 96, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Downsample(
          (op): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (x_upd): Downsample(
          (op): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=192, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 96, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Identity()
      )
    )
    (4-5): 2 x TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 96, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Identity()
        (x_upd): Identity()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=192, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 96, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Identity()
      )
    )
    (6): TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 96, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Downsample(
          (op): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (x_upd): Downsample(
          (op): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=192, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 96, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Identity()
      )
    )
    (7): TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 96, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(96, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Identity()
        (x_upd): Identity()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=384, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 192, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Conv2d(96, 192, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (8): TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 192, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Identity()
        (x_upd): Identity()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=384, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 192, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Identity()
      )
    )
    (9): TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 192, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Downsample(
          (op): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (x_upd): Downsample(
          (op): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=384, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 192, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Identity()
      )
    )
    (10-11): 2 x TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 192, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Identity()
        (x_upd): Identity()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=384, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 192, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Identity()
      )
      (1): AttentionBlock(
        (norm): GroupNorm32(32, 192, eps=1e-05, affine=True)
        (qkv): Conv1d(192, 576, kernel_size=(1,), stride=(1,))
        (attention): QKVAttentionLegacy()
        (proj_out): Conv1d(192, 192, kernel_size=(1,), stride=(1,))
      )
    )
    (12): TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 192, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Downsample(
          (op): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (x_upd): Downsample(
          (op): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=384, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 192, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Identity()
      )
    )
    (13): TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 192, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(192, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Identity()
        (x_upd): Identity()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=576, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 288, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Conv2d(192, 288, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): AttentionBlock(
        (norm): GroupNorm32(32, 288, eps=1e-05, affine=True)
        (qkv): Conv1d(288, 864, kernel_size=(1,), stride=(1,))
        (attention): QKVAttentionLegacy()
        (proj_out): Conv1d(288, 288, kernel_size=(1,), stride=(1,))
      )
    )
    (14): TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 288, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Identity()
        (x_upd): Identity()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=576, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 288, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Identity()
      )
      (1): AttentionBlock(
        (norm): GroupNorm32(32, 288, eps=1e-05, affine=True)
        (qkv): Conv1d(288, 864, kernel_size=(1,), stride=(1,))
        (attention): QKVAttentionLegacy()
        (proj_out): Conv1d(288, 288, kernel_size=(1,), stride=(1,))
      )
    )
    (15): TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 288, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Downsample(
          (op): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (x_upd): Downsample(
          (op): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=576, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 288, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Identity()
      )
    )
    (16-17): 2 x TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 288, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Identity()
        (x_upd): Identity()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=576, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 288, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Identity()
      )
      (1): AttentionBlock(
        (norm): GroupNorm32(32, 288, eps=1e-05, affine=True)
        (qkv): Conv1d(288, 864, kernel_size=(1,), stride=(1,))
        (attention): QKVAttentionLegacy()
        (proj_out): Conv1d(288, 288, kernel_size=(1,), stride=(1,))
      )
    )
  )
  (middle_block): TimestepEmbedSequential(
    (0): ResBlock(
      (in_layers): Sequential(
        (0): GroupNorm32(32, 288, eps=1e-05, affine=True)
        (1): SiLU()
        (2): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (h_upd): Identity()
      (x_upd): Identity()
      (emb_layers): Sequential(
        (0): SiLU()
        (1): Linear(in_features=384, out_features=576, bias=True)
      )
      (out_layers): Sequential(
        (0): GroupNorm32(32, 288, eps=1e-05, affine=True)
        (1): SiLU()
        (2): Dropout(p=0, inplace=False)
        (3): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (skip_connection): Identity()
    )
    (1): AttentionBlock(
      (norm): GroupNorm32(32, 288, eps=1e-05, affine=True)
      (qkv): Conv1d(288, 864, kernel_size=(1,), stride=(1,))
      (attention): QKVAttentionLegacy()
      (proj_out): Conv1d(288, 288, kernel_size=(1,), stride=(1,))
    )
    (2): ResBlock(
      (in_layers): Sequential(
        (0): GroupNorm32(32, 288, eps=1e-05, affine=True)
        (1): SiLU()
        (2): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (h_upd): Identity()
      (x_upd): Identity()
      (emb_layers): Sequential(
        (0): SiLU()
        (1): Linear(in_features=384, out_features=576, bias=True)
      )
      (out_layers): Sequential(
        (0): GroupNorm32(32, 288, eps=1e-05, affine=True)
        (1): SiLU()
        (2): Dropout(p=0, inplace=False)
        (3): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (skip_connection): Identity()
    )
  )
  (output_blocks): ModuleList(
    (0-1): 2 x TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 576, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(576, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Identity()
        (x_upd): Identity()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=576, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 288, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Conv2d(576, 288, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): AttentionBlock(
        (norm): GroupNorm32(32, 288, eps=1e-05, affine=True)
        (qkv): Conv1d(288, 864, kernel_size=(1,), stride=(1,))
        (attention): QKVAttentionLegacy()
        (proj_out): Conv1d(288, 288, kernel_size=(1,), stride=(1,))
      )
    )
    (2): TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 576, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(576, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Identity()
        (x_upd): Identity()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=576, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 288, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Conv2d(576, 288, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): AttentionBlock(
        (norm): GroupNorm32(32, 288, eps=1e-05, affine=True)
        (qkv): Conv1d(288, 864, kernel_size=(1,), stride=(1,))
        (attention): QKVAttentionLegacy()
        (proj_out): Conv1d(288, 288, kernel_size=(1,), stride=(1,))
      )
      (2): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 288, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Upsample()
        (x_upd): Upsample()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=576, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 288, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Identity()
      )
    )
    (3-4): 2 x TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 576, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(576, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Identity()
        (x_upd): Identity()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=576, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 288, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Conv2d(576, 288, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): AttentionBlock(
        (norm): GroupNorm32(32, 288, eps=1e-05, affine=True)
        (qkv): Conv1d(288, 864, kernel_size=(1,), stride=(1,))
        (attention): QKVAttentionLegacy()
        (proj_out): Conv1d(288, 288, kernel_size=(1,), stride=(1,))
      )
    )
    (5): TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 480, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(480, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Identity()
        (x_upd): Identity()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=576, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 288, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Conv2d(480, 288, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): AttentionBlock(
        (norm): GroupNorm32(32, 288, eps=1e-05, affine=True)
        (qkv): Conv1d(288, 864, kernel_size=(1,), stride=(1,))
        (attention): QKVAttentionLegacy()
        (proj_out): Conv1d(288, 288, kernel_size=(1,), stride=(1,))
      )
      (2): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 288, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Upsample()
        (x_upd): Upsample()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=576, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 288, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Identity()
      )
    )
    (6): TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 480, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(480, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Identity()
        (x_upd): Identity()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=384, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 192, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Conv2d(480, 192, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): AttentionBlock(
        (norm): GroupNorm32(32, 192, eps=1e-05, affine=True)
        (qkv): Conv1d(192, 576, kernel_size=(1,), stride=(1,))
        (attention): QKVAttentionLegacy()
        (proj_out): Conv1d(192, 192, kernel_size=(1,), stride=(1,))
      )
    )
    (7): TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 384, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(384, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Identity()
        (x_upd): Identity()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=384, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 192, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): AttentionBlock(
        (norm): GroupNorm32(32, 192, eps=1e-05, affine=True)
        (qkv): Conv1d(192, 576, kernel_size=(1,), stride=(1,))
        (attention): QKVAttentionLegacy()
        (proj_out): Conv1d(192, 192, kernel_size=(1,), stride=(1,))
      )
    )
    (8): TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 384, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(384, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Identity()
        (x_upd): Identity()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=384, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 192, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): AttentionBlock(
        (norm): GroupNorm32(32, 192, eps=1e-05, affine=True)
        (qkv): Conv1d(192, 576, kernel_size=(1,), stride=(1,))
        (attention): QKVAttentionLegacy()
        (proj_out): Conv1d(192, 192, kernel_size=(1,), stride=(1,))
      )
      (2): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 192, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Upsample()
        (x_upd): Upsample()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=384, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 192, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Identity()
      )
    )
    (9-10): 2 x TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 384, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(384, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Identity()
        (x_upd): Identity()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=384, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 192, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (11): TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 288, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(288, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Identity()
        (x_upd): Identity()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=384, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 192, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Conv2d(288, 192, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 192, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Upsample()
        (x_upd): Upsample()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=384, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 192, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Identity()
      )
    )
    (12): TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 288, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(288, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Identity()
        (x_upd): Identity()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=192, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 96, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Conv2d(288, 96, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (13): TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 192, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(192, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Identity()
        (x_upd): Identity()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=192, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 96, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Conv2d(192, 96, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (14): TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 192, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(192, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Identity()
        (x_upd): Identity()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=192, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 96, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Conv2d(192, 96, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 96, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Upsample()
        (x_upd): Upsample()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=192, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 96, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Identity()
      )
    )
    (15-17): 3 x TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 192, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(192, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Identity()
        (x_upd): Identity()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=192, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 96, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Conv2d(192, 96, kernel_size=(1, 1), stride=(1, 1))
      )
    )
  )
  (input_blocks_lr): ModuleList(
    (0): TimestepEmbedSequential(
      (0): Conv2d(3, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (1-2): 2 x TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 96, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Identity()
        (x_upd): Identity()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=192, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 96, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Identity()
      )
    )
    (3): TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 96, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Downsample(
          (op): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (x_upd): Downsample(
          (op): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=192, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 96, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Identity()
      )
    )
    (4-5): 2 x TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 96, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Identity()
        (x_upd): Identity()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=192, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 96, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Identity()
      )
    )
    (6): TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 96, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Downsample(
          (op): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (x_upd): Downsample(
          (op): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=192, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 96, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Identity()
      )
    )
    (7): TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 96, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(96, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Identity()
        (x_upd): Identity()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=384, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 192, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Conv2d(96, 192, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (8): TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 192, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Identity()
        (x_upd): Identity()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=384, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 192, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Identity()
      )
    )
    (9): TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 192, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Downsample(
          (op): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (x_upd): Downsample(
          (op): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=384, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 192, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Identity()
      )
    )
    (10-11): 2 x TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 192, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Identity()
        (x_upd): Identity()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=384, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 192, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Identity()
      )
      (1): AttentionBlock(
        (norm): GroupNorm32(32, 192, eps=1e-05, affine=True)
        (qkv): Conv1d(192, 576, kernel_size=(1,), stride=(1,))
        (attention): QKVAttentionLegacy()
        (proj_out): Conv1d(192, 192, kernel_size=(1,), stride=(1,))
      )
    )
    (12): TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 192, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Downsample(
          (op): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (x_upd): Downsample(
          (op): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=384, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 192, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Identity()
      )
    )
    (13): TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 192, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(192, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Identity()
        (x_upd): Identity()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=576, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 288, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Conv2d(192, 288, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): AttentionBlock(
        (norm): GroupNorm32(32, 288, eps=1e-05, affine=True)
        (qkv): Conv1d(288, 864, kernel_size=(1,), stride=(1,))
        (attention): QKVAttentionLegacy()
        (proj_out): Conv1d(288, 288, kernel_size=(1,), stride=(1,))
      )
    )
    (14): TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 288, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Identity()
        (x_upd): Identity()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=576, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 288, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Identity()
      )
      (1): AttentionBlock(
        (norm): GroupNorm32(32, 288, eps=1e-05, affine=True)
        (qkv): Conv1d(288, 864, kernel_size=(1,), stride=(1,))
        (attention): QKVAttentionLegacy()
        (proj_out): Conv1d(288, 288, kernel_size=(1,), stride=(1,))
      )
    )
    (15): TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 288, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Downsample(
          (op): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (x_upd): Downsample(
          (op): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=576, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 288, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Identity()
      )
    )
    (16-17): 2 x TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 288, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Identity()
        (x_upd): Identity()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=576, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 288, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Identity()
      )
      (1): AttentionBlock(
        (norm): GroupNorm32(32, 288, eps=1e-05, affine=True)
        (qkv): Conv1d(288, 864, kernel_size=(1,), stride=(1,))
        (attention): QKVAttentionLegacy()
        (proj_out): Conv1d(288, 288, kernel_size=(1,), stride=(1,))
      )
    )
  )
  (input_blocks_other): ModuleList(
    (0): TimestepEmbedSequential(
      (0): Conv2d(3, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (1-2): 2 x TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 96, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Identity()
        (x_upd): Identity()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=192, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 96, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Identity()
      )
    )
    (3): TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 96, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Downsample(
          (op): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (x_upd): Downsample(
          (op): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=192, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 96, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Identity()
      )
    )
    (4-5): 2 x TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 96, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Identity()
        (x_upd): Identity()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=192, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 96, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Identity()
      )
    )
    (6): TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 96, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Downsample(
          (op): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (x_upd): Downsample(
          (op): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=192, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 96, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Identity()
      )
    )
    (7): TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 96, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(96, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Identity()
        (x_upd): Identity()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=384, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 192, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Conv2d(96, 192, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (8): TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 192, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Identity()
        (x_upd): Identity()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=384, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 192, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Identity()
      )
    )
    (9): TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 192, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Downsample(
          (op): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (x_upd): Downsample(
          (op): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=384, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 192, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Identity()
      )
    )
    (10-11): 2 x TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 192, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Identity()
        (x_upd): Identity()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=384, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 192, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Identity()
      )
      (1): AttentionBlock(
        (norm): GroupNorm32(32, 192, eps=1e-05, affine=True)
        (qkv): Conv1d(192, 576, kernel_size=(1,), stride=(1,))
        (attention): QKVAttentionLegacy()
        (proj_out): Conv1d(192, 192, kernel_size=(1,), stride=(1,))
      )
    )
    (12): TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 192, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Downsample(
          (op): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (x_upd): Downsample(
          (op): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=384, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 192, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Identity()
      )
    )
    (13): TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 192, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(192, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Identity()
        (x_upd): Identity()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=576, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 288, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Conv2d(192, 288, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): AttentionBlock(
        (norm): GroupNorm32(32, 288, eps=1e-05, affine=True)
        (qkv): Conv1d(288, 864, kernel_size=(1,), stride=(1,))
        (attention): QKVAttentionLegacy()
        (proj_out): Conv1d(288, 288, kernel_size=(1,), stride=(1,))
      )
    )
    (14): TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 288, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Identity()
        (x_upd): Identity()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=576, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 288, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Identity()
      )
      (1): AttentionBlock(
        (norm): GroupNorm32(32, 288, eps=1e-05, affine=True)
        (qkv): Conv1d(288, 864, kernel_size=(1,), stride=(1,))
        (attention): QKVAttentionLegacy()
        (proj_out): Conv1d(288, 288, kernel_size=(1,), stride=(1,))
      )
    )
    (15): TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 288, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Downsample(
          (op): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (x_upd): Downsample(
          (op): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=576, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 288, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Identity()
      )
    )
    (16-17): 2 x TimestepEmbedSequential(
      (0): ResBlock(
        (in_layers): Sequential(
          (0): GroupNorm32(32, 288, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (h_upd): Identity()
        (x_upd): Identity()
        (emb_layers): Sequential(
          (0): SiLU()
          (1): Linear(in_features=384, out_features=576, bias=True)
        )
        (out_layers): Sequential(
          (0): GroupNorm32(32, 288, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0, inplace=False)
          (3): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (skip_connection): Identity()
      )
      (1): AttentionBlock(
        (norm): GroupNorm32(32, 288, eps=1e-05, affine=True)
        (qkv): Conv1d(288, 864, kernel_size=(1,), stride=(1,))
        (attention): QKVAttentionLegacy()
        (proj_out): Conv1d(288, 288, kernel_size=(1,), stride=(1,))
      )
    )
  )
  (CBAM_com): CBAM(
    (ca): ChannelAttention(
      (avg_pool): AdaptiveAvgPool2d(output_size=1)
      (max_pool): AdaptiveMaxPool2d(output_size=1)
      (fc1): Conv2d(144, 9, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (relu1): ReLU()
      (fc2): Conv2d(9, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (sigmoid): Sigmoid()
    )
    (sa): SpatialAttention(
      (conv1): Conv2d(2, 1, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
      (sigmoid): Sigmoid()
    )
  )
  (CBAM_dist_1): CBAM(
    (ca): ChannelAttention(
      (avg_pool): AdaptiveAvgPool2d(output_size=1)
      (max_pool): AdaptiveMaxPool2d(output_size=1)
      (fc1): Conv2d(144, 9, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (relu1): ReLU()
      (fc2): Conv2d(9, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (sigmoid): Sigmoid()
    )
    (sa): SpatialAttention(
      (conv1): Conv2d(2, 1, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
      (sigmoid): Sigmoid()
    )
  )
  (CBAM_dist_2): CBAM(
    (ca): ChannelAttention(
      (avg_pool): AdaptiveAvgPool2d(output_size=1)
      (max_pool): AdaptiveMaxPool2d(output_size=1)
      (fc1): Conv2d(144, 9, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (relu1): ReLU()
      (fc2): Conv2d(9, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (sigmoid): Sigmoid()
    )
    (sa): SpatialAttention(
      (conv1): Conv2d(2, 1, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
      (sigmoid): Sigmoid()
    )
  )
  (CBAM_dist_3): CBAM(
    (ca): ChannelAttention(
      (avg_pool): AdaptiveAvgPool2d(output_size=1)
      (max_pool): AdaptiveMaxPool2d(output_size=1)
      (fc1): Conv2d(144, 9, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (relu1): ReLU()
      (fc2): Conv2d(9, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (sigmoid): Sigmoid()
    )
    (sa): SpatialAttention(
      (conv1): Conv2d(2, 1, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
      (sigmoid): Sigmoid()
    )
  )
  (dim_reduction_non_zeros): Sequential(
    (0): Conv2d(576, 288, kernel_size=(1, 1), stride=(1, 1))
    (1): SiLU()
  )
  (conv_common): Sequential(
    (0): Conv2d(288, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): SiLU()
  )
  (conv_distinct): Sequential(
    (0): Conv2d(288, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): SiLU()
  )
  (out): Sequential(
    (0): GroupNorm32(32, 96, eps=1e-05, affine=True)
    (1): SiLU()
    (2): Conv2d(96, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
)
torch.Size([16, 3, 64, 64]) torch.Size([16]) torch.Size([16, 3, 64, 64]) torch.Size([16, 3, 64, 64])
torch.Size([16, 144, 2, 2]) torch.Size([16, 144, 2, 2]) torch.Size([16, 144, 2, 2]) torch.Size([16, 144, 2, 2]) torch.Size([16, 144, 2, 2]) torch.Size([16, 576, 2, 2])
torch.Size([16, 144, 2, 2]) torch.Size([16, 144, 2, 2]) torch.Size([16, 144, 2, 2]) torch.Size([16, 144, 2, 2]) torch.Size([16, 2, 64, 64])

Process finished with exit code 0


'''