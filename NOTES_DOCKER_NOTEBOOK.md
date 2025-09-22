# Dockerfile to containerize the notebook environment

### Specify the base image for the build
FROM python:3.10

### Set the working directory in the container from which all subsequent commands will be run
WORKDIR /notebook

### Copy the requirements file into the container in the working directory (signified by the dot)
COPY requirements-note.txt .

### Install the dependencies specified in the requirements file
RUN pip install --no-cache-dir -r requirements-app.txt 

By not storing the downloaded dependencies in a cache, you keep the docker image as small as possible. Trade off is the loading time with each install.

### Copy the required files into the container in the working directory
COPY Diabetes_predictor_v4.ipynb .

### Expose the port that the streamlit application will run on
EXPOSE 8888

### Define the container's primary executable
ENTRYPOINT ["jupyter", "lab"]

**N.B.** Alternatively "jupyter", "notebook" for jupyter notebook server.

### Specify the commands to run the application (default arguments to the executable defined in ENTRYPOINT)
CMD ["--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]

**N.B.**
--port=8888 is the default port for jupyter
--no-browser tells JupyterLab not to automatically open a web browser when the server starts.  If you don't include this flag, the server will try to open a browser and fail, which could cause an error or an unresponsive process.
--ip=0.0.0.0 configures the server to listen on all available network interfaces. The IP address 0.0.0.0 is a special value that represents all IP addresses on the machine. If you don't use this flag, the server will not be accessible from outside the container, which means you wouldn't be able to access it from your web browser on the host machine.
--allow-root allows the JupyterLab server to run as the root user inside the container. It simplifies the setup and can prevent permission errors if you are performing operations that require elevated privileges, such as installing packages or writing to certain directories. This is common practice in development environments.
