## SCANN

```
apt-get update && apt-get install -y apt-transport-https gnupg curl
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg
mv bazel.gpg /etc/apt/trusted.gpg.d/
echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
apt-get update && apt-get install -y bazel
bazel --version

curl -Lo /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
chmod +x /usr/local/bin/bazel
bazel --version

apt-get install -y libhdf5-dev
pip install --no-binary=h5py h5py

export HDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/serial/
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/hdf5/serial/:$LD_LIBRARY_PATH

git clone https://github.com/google-research/google-research.git
cd google-research/scann

python configure.py

bazel build -c opt --copt=-mavx2 --copt=-mfma //scann/...
```
