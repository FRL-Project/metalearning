#ARG PARENT_IMAGE=rlworkgroup/garage-headless
ARG PARENT_IMAGE=rlworkgroup/garage-nvidia
FROM $PARENT_IMAGE

# install requirements
ADD ./requirements.txt /requirements.txt
RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir -r /requirements.txt

# get licence for mujoco and copy it to correct folder
RUN curl -LJO https://www.roboti.us/file/mjkey.txt
RUN cp mjkey.txt ~/.mujoco/mjkey.txt

# give wirte access to the host project folder
# not sure if this is the right way to give write access
USER root
RUN mkdir /opt/project
RUN chown $USER: /opt/project