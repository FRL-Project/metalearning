BASH_RC="${HOME}/.bashrc"

# Leave a note in ~/.bashrc for the added environment variables
echo -e "\n# Added by the garage/mujoco installer" >> "${BASH_RC}"

# Set up MuJoCo 2.0 (for gym and dm_control)
if [[ ! -d "${HOME}/.mujoco/mujoco200_linux" ]]; then
  echo "Installing mujoco200"
  mkdir -p "${HOME}"/.mujoco
  MUJOCO_ZIP="$(mktemp -d)/mujoco.zip"
  wget https://www.roboti.us/download/mujoco200_linux.zip -O "${MUJOCO_ZIP}"
  unzip -u "${MUJOCO_ZIP}" -d "${HOME}"/.mujoco
  ln -s "${HOME}"/.mujoco/mujoco200_linux "${HOME}"/.mujoco/mujoco200
fi
# dm_control viewer requires MUJOCO_GL to be set to work
echo "export MUJOCO_GL=\"glfw\"" >> "${BASH_RC}"

# Configure MuJoCo as a shared library
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco200/bin"
LD_LIB_ENV_VAR="LD_LIBRARY_PATH=\"\$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco200"
LD_LIB_ENV_VAR="${LD_LIB_ENV_VAR}/bin\""
echo "export ${LD_LIB_ENV_VAR}" >> "${BASH_RC}"


# We need a MuJoCo key to import mujoco_py
if [[ ! -f "${HOME}/.mujoco/mjkey.txt" ]]; then
  echo "Installing key"
  wget https://www.roboti.us/file/mjkey.txt -O "${HOME}"/.mujoco/mjkey.txt
fi
