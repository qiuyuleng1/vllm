import sys
from vllm.entrypoints.openai.cli_args import make_arg_parser
import xfastertransformer

# for logits send
from mpi4py import MPI
import numpy as np

def parse_args():
    parser = make_arg_parser()
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args.model)
    print(args.dtype)
    print(args.kv_cache_dtype)

    model = xfastertransformer.AutoModel.from_pretrained(
        args.model, dtype=args.dtype, kv_cache_dtype=args.kv_cache_dtype
    )
    print("=============slave===============")
    # if model.rank == 0:
    #     print("[ERROR] llm.entrypoints.slave shouldn't be Master!")
    #     sys.exit(0)
        
    comm = MPI.COMM_WORLD

    while True:
        with open("output.txt", "a") as f:  
            f.write("=============slave===============")
            f.flush()
            f.write(f"model rank: {model.rank}, color: {model.color}, section: {model.section}")
            f.flush()
            model.set_input_cb()
            f.write("finish set_input_cb")
            f.flush()
            logits = model.forward_cb()
            f.write("\n logits: " + np.array2string(logits.detach().cpu().numpy()) + "\n") 

            if model.color == 1:
                print_info = "开始发送数据"
                f.write(print_info + "\n")  

                comm.send(logits, dest=0)  

                print_info = "数据已发送"
                f.write(print_info + "\n")  #直接写入你要的字符串
                
            model.free_seqs()