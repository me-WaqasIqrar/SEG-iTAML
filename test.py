import torch

def check_cuda_availability():
    if torch.cuda.is_available():
        print("CUDA is available. GPU can be used.")
    else:
        print("CUDA is not available. Only CPU can be used.")

if __name__ == "__main__":
    check_cuda_availability()
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Capability: {torch.cuda.get_device_capability(0)}")