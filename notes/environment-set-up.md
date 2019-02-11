# Environment set up

*The instructions below are intentionally verbose as they try to explain the reasoning behind our choice of environment set up and to explain what each command we are asking you to run does. If you are already confident using bash, Conda environments and Git you may wish to instead use the much shorter [minimal set-up instructions](#minimal-set-up-instructions-for-dice) at the end which skip the explanations.*

In this course we will be using [Python 3](https://www.python.org/) for all the labs and coursework assignments. In particular we will be making heavy use of the numerical computing libraries [NumPy](http://www.numpy.org/) and [SciPy](http://www.scipy.org/), and the interactive notebook application [Jupyter](http://jupyter.org/).

A common headache in software projects is ensuring the correct versions of all dependencies are available on the current development system. Often you may be working on several distinct projects simultaneously each with its own potentially conflicting dependencies on external libraries. Additionally you may be working across multiple different machines (for example a personal laptop and University computers) with possibly different operating systems. Further, as is the case in Informatics on DICE, you may not have root-level access to a system you are working on and so not be able to install software at a system-wide level and system updates may cause library versions to be changed to incompatible versions.

One way of overcoming these issues is to use project-specific *virtual environments*. In this context a virtual environment is an isolated development environment where the external dependencies of a project can be installed and managed independent of the system-wide versions (and those of the environments of other projects).

There are several virtual environment solutions available in the Python eco-system, including the native [pyvenv](https://docs.python.org/3/library/venv.html) in Python 3 and the popular [virtualenv](https://virtualenv.pypa.io/en/stable/). Also related is [pip](https://pip.pypa.io/en/stable/) a Python package manager natively included in Python 2.7.9 and above.

Here we will instead use the environment capabilities of the [Conda](http://conda.pydata.org/docs/) package management system. Unlike pip and virtualenv/pyvenv, Conda is not limited to managing Python packages but is a language and platform agnostic package manager. Both NumPy and SciPy have many non-Python external dependencies and their performance is very dependent on correctly linking to optimised linear algebra libraries.

Conda can handle installation of the Python libraries we will be using and all their external dependencies, in particular allowing easy installation of [optimised numerical computing libraries](https://docs.continuum.io/mkl-optimizations/). Further Conda can easily be installed on Linux, OSX and Windows systems meaning if you wish to set up an environment on a personal machine as well this should be easy to do whatever your operating system of choice is.

There are several options available for installing Conda on a system. Here we will use the Python 3 version of [Miniconda](http://conda.pydata.org/miniconda.html), which installs just Conda and its dependencies. An alternative is to install the [Anaconda Python distribution](https://docs.continuum.io/anaconda/), which installs Conda and a large selection of popular Python packages. As we will require only a small subset of these packages we will use the more barebones Miniconda to avoid eating into your DICE disk quota too much, however if installing on a personal machine you may wish to consider Anaconda if you want to explore other Python packages.

## Installing Miniconda

We provide instructions here for getting an environment with all the required dependencies running on computers running 
the School of Informatics [DICE desktop](http://computing.help.inf.ed.ac.uk/dice-platform). The same instructions 
should be able to used on other Linux distributions such as Ubuntu and Linux Mint with minimal adjustments.

For those wishing to install on a personal Windows or OSX machine, the initial instructions for setting up Conda will 
differ slightly - you should instead select the relevant installer for your system from [here](http://conda.pydata.org/miniconda.html) and following the corresponding installation instructions from [here](http://conda.pydata.org/docs/install/quick.html). After Conda is installed the [remaining instructions](#creating-the-conda-environment) should be broadly the same across different systems.

*Note: Although we are happy for you to additionally set up an environment on a personal machine, you should still set up a DICE environment now as this will make sure you are able to use shared computing resources later in the course. Also although we have tried to note when the required commands will differ on non-DICE systems, these instructions have only been tested on DICE and we will not be able to offer any support in labs on getting set up on a non-DICE system.*

---

Open a bash terminal (`Applications > Terminal` on DICE).

We first need to download the latest 64-bit Python 3 Miniconda install script:

```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

This uses `wget` a command-line tool for downloading files.

Now run the install script:

```
bash Miniconda3-latest-Linux-x86_64.sh
```

You will first be asked to review the software license agreement. Assuming you choose to agree, you will then be asked 
to choose an install location for Miniconda. The default is to install in the root of your home directory 
`~/miniconda3`. We recommend going with this default unless you have a particular reason to do otherwise.

You will then be asked whether to prepend the Miniconda binaries directory to the `PATH` system environment variable 
definition in `.bashrc`. As the DICE bash start-up mechanism differs from the standard set up 
([details here](http://computing.help.inf.ed.ac.uk/dice-bash)), on DICE you should respond `no` here as we will set up the addition to `PATH` manually in the next step. On other Linux distributions you may choose to accept the default.

On DICE, append the Miniconda binaries directory to `PATH` in manually in `~/.benv` using

```
echo "export PATH=\""\$PATH":$HOME/miniconda3/bin\"" >> ~/.benv
```

For those who this appears a bit opaque to and want to know what is going on see here <sup id="a1">[1](#f1)</sup>.

We now need to `source` the updated `~/.benv` so that the `PATH` variable in the current terminal session is updated:

```
source ~/.benv
```

From the next time you log in all future terminal sessions should have the updated `PATH` loaded by default.

## Creating the Conda environment

You should now have a working Conda installation. If you run

```
conda --help
```
from a terminal you should see the Conda help page displayed. If you get a `No command 'conda' found` error you should check you have set up your `PATH` variable correctly (you can get a demonstrator to help you do this).

Assuming Conda is working, we will now create our Conda environment:

```
conda create -n mlp python=3
```

This bootstraps a new Conda environment named `mlp` with a minimal Python 3 install. You will be presented with a 'package plan' listing the packages to be installed and asked whether to proceed: type `y` then enter.

We will now *activate* our created environment:

```
source activate mlp
```

or on Windows only

```
activate mlp
```

When a environment is activated its name will be prepended on to the prompt which should now look something like `(mlp) [machine-name]:~$` on DICE.

**You need to run this `source activate mlp` command every time you wish to activate the `mlp` environment in a terminal (for example at the beginning of each lab)**. When the environment is activated, the environment will be searched first when running commands so that e.g. `python` will launch the Python interpreter installed locally in the `mlp` environment rather than a system-wide version.

If you wish to deactivate an environment loaded in the current terminal e.g. to launch the system Python interpreter, you can run `source deactivate` (just `deactivate` on Windows).

We will now install the dependencies for the course into the new environment:

```
conda install numpy scipy matplotlib jupyter
```

Again you will be given a list of the packages to be installed and asked to confirm whether to proceed. Enter `y` then wait for the packages to install (this should take around five minutes). In addition to Jupyter, NumPy and SciPy which we have already mentioned, we are also installing [matplotlib](http://matplotlib.org/) a plotting and visualisation library.

Once the installation is finished, to recover some disk space we can clear the package tarballs Conda just downloaded:

```
conda clean -t
```

These tarballs are usually cached to allow quicker installation into additional environments however we will only be using a single environment here so there is no need to keep them on disk.

## Getting the course code and a short introduction to Git

The next step in getting our environment set up will be to download the course code. This is available in a Git repository on Github:

https://github.com/CSTR-Edinburgh/mlpractical

[Git](https://git-scm.com/) is a distributed version control system and [Github](https://github.com) a popular site for hosting Git repositories. We will be using Git to distribute the code for all the labs and assignments. We will explain all the necessary `git` commands as we go, though those new to Git may find [this concise guide by Roger Dudler](http://rogerdudler.github.io/git-guide/) or [this slightly longer one from Atlassian](https://www.atlassian.com/git/tutorials/) useful.

---

***Non-DICE systems only:***

Git is installed by default on DICE desktops. If you are running a system which does not have Git installed, you can use Conda to install it in your environment using:

```
conda install git
```

---

We will now go over the process of [cloning](https://www.atlassian.com/git/tutorials/setting-up-a-repository/git-clone) a local copy of the `mlpractical` repository.

---
**Confident Git users only:**

For those who have their own Github account and are confident Git users, you may wish to consider instead [creating a private fork](http://stackoverflow.com/a/30352360) of the `CSTR-Edinburgh/mlpractical` repository on Github. This is not required for the course, however it will allow you to push your local commits to Github making it easier to for example sync your work between DICE computers and a personal machine.

**Note you should NOT create a public fork using the default forking mechanism on Github as this will make any commits you push to the fork publicly available which creates a risk of plagiarism.**

If you are already familiar with Git you may wish to skip over the explanatory sections below, though you should read [the section on how we will use branches to separate the code for different labs](#branching-explanation).

---

By default we will assume here you are cloning to your home directory however if you have an existing system for organising your workspace feel free to keep to that. **If you clone the repository to a path other than `~/mlpractical` however you will need to adjust all references to `~/mlpractical` in the commands below accordingly.**


To clone the `mlpractical` repository to the home directory run

```
git clone https://github.com/CSTR-Edinburgh/mlpractical.git ~/mlpractical
```

This will create a new `mlpractical` subdirectory with a local copy of the repository in it. Enter the directory and list all its contents, including hidden files, by running:

```
cd ~/mlpractical
ls -a  # Windows equivalent: dir /a
```

For the most part this will look much like any other directory, with there being the following three non-hidden sub-directories:

  * `data`: Data files used in the labs and assignments.
  * `mlp`: The custom Python package we will use in this course.
  * `notebooks`: The Jupyter notebook files for each lab and coursework.

Additionally there exists a hidden `.git` subdirectory (on Unix systems by default files and directories prepended with a period '.' are hidden). This directory contains the repository history database and various configuration files and references. Unless you are sure you know what you are doing you generally should not edit any of the files in this directory directly. Generally most configuration options can be enacted more safely using a `git config` command.


For instance to globally set the user name and email used in commits you can run:

```
git config --global user.name "[your name]"
git config --global user.email "[matric-number]@sms.ed.ac.uk"
```

*Note this is meant as an example of a `git config` command - you do not need to run this command though there is no harm in doing so.*

From the  `~/mlpractical` directory if you now run:

`git status`

a status message containing information about your local clone of the repository should be displayed.

Providing you have not made any changes yet, all that will be displayed is the name of the current *branch* (we will explain what a branch is to those new to Git in a little while), a message that the branch is up to date with the remote repository and that there is nothing to commit in the working directory.

The two key concepts you will need to know about Git for this course are *commits* and *branches*.

A *commit* in Git is a snapshot of the state of the project. The snapshots are recorded in the repository history and allow us to track changes to the code over time and rollback changes if necessary. In Git there is a three stage process to creating a new commit.

  1. The relevant edits are made to files in the working directory and any new files created.

  2. The files with changes to be committed (including any new files) are added to the *staging area* by running:

  ```
  git add file1 file2 ...
  ```

  3. Finally the *staged changes* are used to create a new commit by running

  ```
  git commit -m "A commit message describing the changes."
  ```

This writes the staged changes as a new commit in the repository history. We can see a log of the details of previous commits by running:

```
git log
```

Although it is not a requirement of the course for you to make regular commits of your work, we strongly recommend you do as it is a good habit to get into and will make recovery from accidental deletions etc. much easier.

The other key Git concept you will need to know about are *branches*. A branch in Git represents an independent line of development of a project. When a repository is first created it will contain a single branch, named `master` by default. Commits to this branch form a linear series of snapshots of the project.

A new branch is created from a commit on an existing branch. Any commits made to this new branch then evolve as an independent and parallel line of changes - that is commits to the new branch will not affect the old branch and vice versa.

A typical Git workflow in a software development setting would be to create a new branch whenever making changes to a project, for example to fix a bug or implement a new feature. These changes are then isolated from the main code base allowing regular commits without worrying about making unstable changes to the main code base. Key to this workflow is the ability to *merge* commits from a branch into another branch, e.g. when it is decided a new feature is sufficiently developed to be added to the main code base. Although merging branches is key aspect of using Git in many projects, as dealing with merge conflicts when two branches both make changes to same parts of files can be a somewhat tricky process, we will here generally try to avoid the need for merges.

<p id='branching-explanation'>We will therefore use branches here in a slightly non-standard way. The code for each week's lab and for each of the assignments will be maintained in a separate branch. This will allow us to stage the release of the notebooks and code for each lab and assignment while allowing you to commit the changes you make to the code each week without having to merge those changes when new code is released. Similarly this structure will allow us to release updated notebooks from previous labs with proposed solutions without overwriting your own work.</p>

To list the branches present in the local repository, run:

```
git branch
```

This will display a list of branches with a `*` next to the current branch. To switch to a different existing branch in the local repository run

```
git checkout branch-name
```

This will change the code in the working directory to the current state of the checked out branch. Any files added to the staging area and committed will then create a new commit on this branch.

You should make sure you are on the first lab branch now by running:

```
git checkout mlp2017-8/lab1
```

## Installing the `mlp` Python package

In your local repository we noted above the presence of a `mlp` subdirectory. This contains the custom Python package implementing the NumPy based neural network framework we will be using in this course.

In order to make the modules in this package available in your environment we need install it. A [setuptools](https://setuptools.readthedocs.io/en/latest/) `setup.py` script is provided in the root of the `mlpractical` directory for this purpose.

The standard way to install a Python package using a `setup.py` script is to run `python setup.py install`. This creates a copy of the package in the `site-packages` directory of the currently active Python environment.

As we will be updating the code in the `mlp` package during the course of the labs this would require you to re-run  `python setup.py install` every time a change is made to the package. Instead therefore you should install the package in development mode by running:

```
python setup.py develop
```

Instead of copying the package, this will instead create a symbolic link to the copy in the local repository. This means any changes made will be immediately available without the need to reinstall the package.

---

**Aside on importing/reloading Python modules:**

Note that after the first time a Python module is loaded into an interpreter instance, using for example:

```
import mlp
```

Running the `import` statement any further times will have no effect even if the underlying module code has been changed. To reload an already imported module we instead need to use the [`reload`](https://docs.python.org/2.7/library/functions.html#reload) function, e.g.

```
reload(mlp)
```

**Note: To be clear as this has caused some confusion in previous labs the above `import ...` / `reload(...)` statements should NOT be run directly in a bash terminal. They are examples Python statements - you could run them in a terminal by first loading a Python interpreter using:**

```
python
```

**however you do not need to do so now. This is meant as information to help you later when importing modules as there was some confusion last year about the difference between `import` and `reload`.**

---

## Adding a data directory variable to the environment

We observed previously the presence of a `data` subdirectory in the local repository. This directory holds the data files that will be used in the course. To enable the data loaders in the `mlp` package to locate these data files we need to set a `MLP_DATA_DIR` environment variable pointing to this directory.

Assuming you used the recommended Miniconda install location and cloned the `mlpractical` repository to your home directory, this variable can be automatically defined when activating the environment by running the following commands (on non-Windows systems):

```
cd ~/miniconda3/envs/mlp
mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d
echo -e '#!/bin/sh\n' >> ./etc/conda/activate.d/env_vars.sh
echo "export MLP_DATA_DIR=$HOME/mlpractical/data" >> ./etc/conda/activate.d/env_vars.sh
echo -e '#!/bin/sh\n' >> ./etc/conda/deactivate.d/env_vars.sh
echo 'unset MLP_DATA_DIR' >> ./etc/conda/deactivate.d/env_vars.sh
export MLP_DATA_DIR=$HOME/mlpractical/data
```

And on Windows systems (replacing the `[]` placeholders with the relevant paths):

```
cd [path-to-conda-root]\envs\mlp
mkdir .\etc\conda\activate.d
mkdir .\etc\conda\deactivate.d
@echo "set MLP_DATA_DIR=[path-to-local-repository]\data" >> .\etc\conda\activate.d\env_vars.bat
@echo "set MLP_DATA_DIR="  >> .\etc\conda\deactivate.d\env_vars.bat
set MLP_DATA_DIR=[path-to-local-repository]\data
```

## Loading the first lab notebook

Your environment is now all set up so you can move on to the introductory exercises in the first lab notebook.

One of the dependencies you installed in your environment earlier was Jupyter. Jupyter notebooks allow combining formatted text with runnable code cells and visualisation of the code output in an intuitive web application interface. Although originally specific to Python (under the previous moniker IPython notebooks) the notebook interface has now been abstracted making them available to a wide range of languages.

There will be a Jupyter notebook available for each lab and assignment in this course, with a combination of explanatory sections for you to read through which will complement the material covered in lectures, as well as series of practical coding exercises to be written and run in the notebook interface. The first lab notebook will cover some of the basics of the notebook interface.

To open a notebook, you first need to launch a Jupyter notebook server instance. From within the `mlpractical` directory containing your local copy of the repository (and with the `mlp` environment activated) run:

```
jupyter notebook
```

This will start a notebook server instance in the current terminal (with a series of status messages being streamed to the terminal output) and launch a browser window which will load the notebook application interface.

By default the notebook interface will show a list of the files in the directory the notebook server was launched from when first loaded. If you click on the `notebooks` directory in this file list, a list of files in this directory should then be displayed. Click the `01_Introduction.ipynb` entry to load the first notebook.

# Minimal set-up instructions for DICE

Below are instructions for setting up the environment without additional explanation. These are intentionally terse and if you do not understand what a particular command is doing you might be better following the more detailed instructions above which explain each step.

---

Start a new bash terminal. Download the latest 64-bit Python 2.7 Miniconda install script:

```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

Run the install script:

```
bash Miniconda3-latest-Linux-x86_64.sh
```

Review the software license agreement and choose whether to accept. Assuming you accept, you be asked to choose an install location for Miniconda. The default is to install in the root of your home directory `~/miniconda3`. We will assume below you have used this default. **If you use a different path you will need to adjust the paths in the commands below to suit.**

You will then be asked whether to prepend the Miniconda binaries directory to the `PATH` system environment variable definition in `.bashrc`. You should respond `no` here as we will set up the addition to `PATH` manually in the next step.

Append the Miniconda binaries directory to `PATH` in manually in `~/.benv`:
```
echo "export PATH=\""\$PATH":$HOME/miniconda3/bin\"" >> ~/.benv
```

`source` the updated `~/.benv`:

```
source ~/.benv
```

Create a new `mlp` Conda environment:

```
conda create -n mlp python=3
```

Activate our created environment:

```
source activate mlp
```

Install the dependencies for the course into the new environment:

```
conda install numpy scipy matplotlib jupyter
```

Clear the package tarballs Conda just downloaded:

```
conda clean -t
```

Clone the course repository to your home directory:

```
git clone https://github.com/CSTR-Edinburgh/mlpractical.git ~/mlpractical
```

Make sure we are on the first lab branch

```
cd ~/mlpractical
git checkout mlp2017-8/lab1
```

Install the `mlp` package in the environment in develop mode

```
python ~/mlpractical/setup.py develop
```

Add an `MLP_DATA_DIR` variable to the environment

```
cd ~/miniconda3/envs/mlp
mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d
echo -e '#!/bin/sh\n' >> ./etc/conda/activate.d/env_vars.sh
echo "export MLP_DATA_DIR=$HOME/mlpractical/data" >> ./etc/conda/activate.d/env_vars.sh
echo -e '#!/bin/sh\n' >> ./etc/conda/deactivate.d/env_vars.sh
echo 'unset MLP_DATA_DIR' >> ./etc/conda/deactivate.d/env_vars.sh
export MLP_DATA_DIR=$HOME/mlpractical/data
```

Environment is now set up. Load the notebook server from `mlpractical` directory

```
cd ~/mlpractical
jupyter notebook
```

and then open the first lab notebook from the `notebooks` directory.


---

<b id="f1">[1]</b> The `echo` command causes the following text to be streamed to an output (standard terminal output by default). Here we use the append redirection operator `>>` to redirect the `echo` output to a file `~/.benv`, with it being appended to the end of the current file. The text actually added is `export PATH="$PATH:[your-home-directory]/miniconda/bin"` with the `\"` being used to escape the quote characters. The `export` command defines system-wide environment variables (more rigorously those inherited by child shells) with `PATH` being the environment variable defining where `bash` searches for executables as a colon-seperated list of directories. Here we add the Miniconda binary directory to the end of the current `PATH` definition. [↩](#a1)
