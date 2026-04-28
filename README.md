# Prerequisites: 

- Make sure you have Python on your computer. This simulation was built on Python 3.14. 
- To ensure that packages stay isolated to this repo, start up a virtual environment to install the necessary prerequisite packages. 
- Run the following command: 

python -m venv .venv

- The virtual environment is now created. Now activate it,

.venv\Scripts\activate

- Then install the packages,

python -m pip install -r 'requirements.txt'

You should now have the necessary packages required to use the simulation. 

# How to run the program: 

- Ensure that you are always in the virtual environment before you run the simulation. Otherwise, you will not have access to the prerequisite python packages, which means the simulation fails. 

- So if you are not in a virtual environment, do this again (from the repo's root):

.venv\Scripts\activate

- Then start the backend python server:

python sim_server.py 

- And then start the frontend HTML server: 

python -m http.server 8080

Your simulation should now be running. Navigate to results\array_explorer.html and start playing with the simulation. 

If you get any errors relating to processes failing to start because one already exists, that means you do not need to perform the above steps. Unless you made changes to the python files performing backend functions, any frontend HTML changes should be served immediately. 
