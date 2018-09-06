How to set-up a Jupyter kernel running Docker
=============================================

To run the notebooks in this directory, or any notebook that uses the D3M image, you need to create a custom kernel definition that will run a Docker container.

To do that, you need to create a directory `_d3m_docker` in `~/.local/share/jupyter/kernels`, and put the kernel definition JSON inside a new file `kernel.json`:

Contents of `~/.local/share/jupyter/kernels/_d3m_docker/kernel.json`:

```json
{
 "argv": [
  "sh",
  "-c",
  "docker run -i --rm --net=host -v $HOME/.local/share/jupyter:$HOME/.local/share/jupyter -v /home/remram/Documents/programming/d3m/data:/d3m/data registry.gitlab.com/vida-nyu/d3m/ta2:devel sh -c \"echo \\\"\\$1\\\" >/jupyter-conn.json && /usr/bin/python3 -m ipykernel_launcher -f /jupyter-conn.json\" -- \"$(cat $1)\"",
  "--",
  "{connection_file}"
 ],
 "display_name": "_d3m_docker",
 "language": "python"
}
```

You might need to update the path to the D3M datasets.
