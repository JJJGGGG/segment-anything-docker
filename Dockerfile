FROM python:3.9.18

WORKDIR /code

# Download SAM Model

RUN mkdir /code/sam_images
RUN curl -o /code/sam_images/sam_vit_l_0b3195.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

# Install the requirements and libraries

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN apt update -y
RUN apt install libgl1-mesa-glx -y

# Copy the app code

COPY ./app /code/app

# Startup command

CMD ["uvicorn", "app.main:app", "--host", "::", "--port", "80"]
