# Running Jupyter notebooks over SSH

Below is a guide for how to start a Jupyter notebook server remotely on one of the shared-use `student.compute` servers and to connect to it on a local machine by port-forwarding over SSH. It is assumed you already have a SSH client set up on the machine you are connecting from and that you are familiar with how to use SSH. These instructions have been written for use with a SSH client running within a terminal session - although it may be possible to replicate the relevant commands within a GUI based SSH client, you will need to figure out how to do this yourself. They were written and tested on Ubuntu 14.04 and no attempt has been made to test them on other operating systems.

## Securing your notebook server

Before running a Jupyter notebook server instance on one of the shared compute servers you **must** make sure you have secured your server by configuring it to use a password and to communicate that password between the browser client and server by secure HTTP. This can be done on by running the `secure-notebook-server.sh` bash script provided in the `scripts` directory of the `mlpractical` repository. You can either do this when logged on to DICE in one of the labs or after connecting to DICE remotely over SSH as described below.

To run the script, in a DICE terminal enter the `mlpractical` repository directory and run
```
bash scripts/secure-notebook-server.sh
```
As this script creates a self-signed certificate to set up the secure HTTP encrypted communication between the browser and server, you will be shown a security warning when you load up the URL the notebooks are being served on.

If you want to manually secure the notebook server yourself or to create a certificate which will stop the security warnings appearing you can refer to the [relevant official Jupyter documentation page](http://jupyter-notebook.readthedocs.io/en/latest/public_server.html).

## Connecting to a remote `student.compute` server over SSH

To start an SSH session, open a terminal window and run

```
ssh [dice-username]@student.ssh.inf.ed.ac.uk
```

If this is this is the first time you have logged on to the SSH gateway server from this computer you will be asked to confirm you wish to connect and a ECDSA key fingerprint printed. You can check this against the reference values on the [school help pages](http://computing.help.inf.ed.ac.uk/external-login).

You will then be asked to enter your password. This is the same password you usually use to log on to DICE.

Assuming you enter the correct password, you will at this point be logged in to the SSH *gateway server*. As the message printed when you log in points out this is intended only for accessing the Informatics network externally and you should **not** attempt to work on this server. You should log in to one of the `student.compute` shared-use servers by running

```
ssh student.compute
```

You should now be logged on to one of the shared-use compute servers. The name of the server you are logged on to will appear at the bash prompt e.g.

```
ashbury:~$
```

You will need to know the name of the remote server you are using later on.

## Starting a notebook server on the remote computer

You should now activate your `mlp` Conda environment by running

```
source activate mlp
```

Now move in to the `mlpractical` local repository directory e.g. by running

```
cd ~/mlpractical
```

if you chose the default of putting the repository in your home directory.

We will now launch a notebook server on the remote compute-server. There are two key differences in the command we use to do this compared to how we usually start up a server on a local machine. First as the server will be running remotely you should set the `--no-browser` option as this will prevent the remote server attempting open a browser to connect to the notebook server.

Secondly we will prefix the command with `nice`. `nice` is a shell command which alters the scheduling priority of the process it is used to start. Its important to use `nice` when running on the shared `student.compute` servers to make sure they remain usable by all of the students who need to run jobs on them. You can set a priority level between 10 (highest priority) and 19 (lowest priority) using the `-n` argument. Running the command below will start up a notebook server at the lowest priority level.

```
nice -n 19 jupyter notebook --no-browser
```

Once the notebook server starts running you should take note of the port it is being served on as indicated in the `The Jupyter Notebook is running at: https://localhost:[port]/` message.

## Forwarding a connection to the notebook server over SSH

Now that the notebook server is running on the remote server you need to connect to it on your local machine. We will do this by forwarding the port the notebook server is being run on over SSH to you local machine. As all external connections from outside the `inf.ed.ac.uk` domain have to go via the SSH gateway server we need to go via this gateway server.

In a **new terminal window / tab** run the command below with the `[...]` placeholders substituted with the appropriate values to securely forward the specified port on the remote server to your local machine and bind it to a local port. You should choose `[remote-port]` to be the port the notebook server is running on on the remote server, `[local-port]` to be a currently unused port on your local machine and `[remote-server-name]` to be the host name of the remote server the notebook server is being run on.

```
ssh -N -o ProxyCommand="ssh -q [dice-username]@student.ssh.inf.ed.ac.uk nc [remote-server-name] 22" \
  -L [local-port]:localhost:[remote-port] [dice-username]@[remote-server-name]
```

You will be asked to enter your (DICE) password twice, once to log on to the gateway server and a second time to log on to the remote compute server.

Assuming you enter your password both times correctly, the remote port will now be getting forwarded to the specified local port on your computer. If you now open up a browser on your computer and go to `https://localhost:[local-port]` you should (potentially after seeing a security warning about the self-signed certicate) now asked to enter the notebook server password you specified earlier. Once you enter this password you should be able to access the notebook dashboard and open and edit notebooks as you usually do in labratories.

When you are finished working you should both close down the notebook server by entering `Ctrl+C` twice in the terminal window the SSH session you used to start up the notebook server is running and halt the port forwarding command by entering `Ctrl+C` in the terminal it is running in.
