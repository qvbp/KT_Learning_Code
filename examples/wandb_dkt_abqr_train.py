import argparse
from wandb_train import main
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="bridge2algebra2006")  
    parser.add_argument("--model_name", type=str, default="dkt_abqr")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model/dkt_abqr")
    # parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=3047)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.4)
    
    parser.add_argument("--emb_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=2e-3)

    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--add_uuid", type=int, default=0)
    
    args = parser.parse_args()

    params = vars(args)
    print("+"*100)

    main(params)


