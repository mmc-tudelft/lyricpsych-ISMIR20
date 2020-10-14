# clone the musixmatch-sdk from github
git clone https://github.com/musixmatch/musixmatch-sdk.git

# move into the repo
cd musixmatch-sdk/dist

# unzip the python sdk distribution
unzip python-client-generated.zip

# reach to the client source and install it to the environment
cd python-client
python setup.py install --user

# remove the repo clone
cd ../../../
rm -rf musixmatch-sdk/

# install other requirements
pip install -r requirements.txt
