Tests out potential of split learning and PeFT methods, Only Util, server and client files are required as of now in the main directory to test it out.
Running: Run the server through `python3 server.py`
For client its strictly needed to have multiple GPUs then use `torchrun --nproc_per_node=2 client.py --server_url "http://127.0.0.1:8000"` Where nproc_per_node refers to the number of GPUs
To load a saved model, just run load.py with the saved model being in saved dir.
