# create a dockerfile to containerize the app.py environemnt

# Specify the base image for the build
FROM python:3.11

# Set the working directory in the container from which all subsequent commands will be run
WORKDIR /app

# Copy the requirements file into the container in the working directory (signified by the dot)
COPY requirements-app.txt .

# Install the dependencies specified in the requirements file
RUN pip install -r requirements-app.txt

# Copy the required files into the container in the working directory
COPY app.py .

# Expose the port that the streamlit application will run on
EXPOSE 8501

# Define the container's primary executable
ENTRYPOINT ["streamlit"]

# Specify the commands to run the application (default arguments to the executable defined in ENTRYPOINT)
CMD ["run", "app.py"]