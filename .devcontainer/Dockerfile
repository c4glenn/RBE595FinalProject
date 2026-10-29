# ros-base is published natively for both amd64 and arm64 (unlike osrf/ros's
# desktop-full images, which are amd64-only). Installing the desktop metapackage
# via apt on top keeps everything native on Apple Silicon instead of emulated.
FROM ros:jazzy-ros-base

ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create a non-root user so files created in the container are usable on the host.
# The base image already ships a UID/GID 1000 "ubuntu" user, so rename it instead
# of failing on a collision.
RUN apt-get update && apt-get install -y sudo \
    && if getent passwd $USER_UID >/dev/null; then \
         existing_user=$(getent passwd $USER_UID | cut -d: -f1); \
         [ "$existing_user" != "$USERNAME" ] && usermod -l $USERNAME -m -d /home/$USERNAME $existing_user; \
       else \
         useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME; \
       fi \
    && if getent group $USER_GID >/dev/null; then \
         existing_group=$(getent group $USER_GID | cut -d: -f1); \
         [ "$existing_group" != "$USERNAME" ] && groupmod -n $USERNAME $existing_group; \
       else \
         groupadd --gid $USER_GID $USERNAME; \
       fi \
    && usermod -s /bin/bash $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Desktop tools (RViz2, Gazebo, demos) + common dev tools, colcon, rosdep
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-desktop-full \
    ros-jazzy-moveit \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    nano \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

RUN rosdep update || true

# Source ROS2 setup automatically for interactive shells
RUN echo "source /opt/ros/jazzy/setup.bash" >> /home/$USERNAME/.bashrc \
    && echo "[ -f /workspace/underlay_ws/install/setup.bash ] && source /workspace/underlay_ws/install/setup.bash" >> /home/$USERNAME/.bashrc \
    && echo "[ -f /workspace/workspace/install/setup.bash ] && source /workspace/workspace/install/setup.bash" >> /home/$USERNAME/.bashrc

USER $USERNAME
WORKDIR /workspace
