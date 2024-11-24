import argparse
from wandb_train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="statics2011")
    parser.add_argument("--model_name", type=str, default="cakt")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    # parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--d_model", type=int, default=256)  # Transformer 模型的维度。例如，256 表示模型的维度为 256。
    # parser.add_argument("--d_ff", type=int, default=512)  # 前馈网络的维度。例如，512 表示前馈网络的维度为 512。
    parser.add_argument("--kernel_size1", type=int, default=1)
    parser.add_argument("--kernel_size2", type=int, default=3)
    parser.add_argument("--num_attn_heads", type=int, default=8)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--add_uuid", type=int, default=0)

    args = parser.parse_args()

    params = vars(args)
    main(params)


