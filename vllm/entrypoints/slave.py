import sys
from vllm.entrypoints.openai.cli_args import make_arg_parser
import xfastertransformer

# for logits send
from mpi4py import MPI
import numpy as np
import os
import torch

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
        with open("slave_output.txt", "a") as f:  
            f.write("=============slave===============")
            f.flush()
            f.write(f"model rank: {model.rank}, color: {model.color}, section: {model.section}")
            f.flush()
            model.set_input_cb()
            f.write("finish set_input_cb")
            f.flush()
            logits = model.forward_cb()
            f.write("\n logits: " + np.array2string(logits.detach().cpu().numpy()) + "\n")
            f.flush()
            
            # xft_pipeline_stage = int(os.environ.get('XFT_PIPELINE_STAGE'))
            # if model.color == xft_pipeline_stage-1 and model.rank == 0:
            #     torch.save(logits, 'pp3_logits.pt') 
            
            # ==========================  测试不处理output，会不会堵塞 ===============================
            xft_pipeline_stage = int(os.environ.get('XFT_PIPELINE_STAGE'))
            
            if model.color == xft_pipeline_stage-1 and model.rank == 0:  # tp+pp时，只让rank 0 发送logits，因为logits是shared memory，谁发都一样
                print_info = "开始发送数据"
                f.write(print_info + "\n")  

                comm.send(logits, dest=0)  

                print_info = "数据已发送"
                f.write(print_info + "\n")  #直接写入你要的字符串
                f.flush()
            # ======================================================================================
                
            model.free_seqs()