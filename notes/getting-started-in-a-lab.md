# Getting started in a lab on DICE computers

Once your [environment is set up](environment-set-up.md), at the beginning of each lab you should be able follow the steps below to get the lab notebook for that session running.

Open a terminal window (`Applications > Terminal`).

We first need to activate our `mlp` Conda environment:

```
source activate mlp
```

We now need to fetch any new code for the lab from the Github repository and create a new branch for this lab's work. First change in to the `mlpractical` repoistory directory (if you cloned the repository to a different directory than the default you will need to adjust the command below accordingly):

```
cd ~/mlpractical
```

If you have not yet commited the changes you made to the current branch in the previous lab you should do so now. You can check if you have changes not yet commited by running `git status`. If there are files with changes to be commited (they will appear in red) you should first add them to the staging area using

```
git add path/to/file1 path/to/file2
```

then commit them with a descriptive commit message using

```
git commit -m "Description of changes e.g. Exercises for first lab notebook."
```

We are now ready to fetch any updated code from the remote repository on Github. This can be done by running

```
git fetch origin
```

This should display a message indicate a new branch has been found and fetched, named `origin/mlp2017-8/lab[n]` where `[n]` is the relevant lab number e.g. `origin/mlp2017-8/lab2` for the second lab.

We now need to create and checkout a new local branch from the remote branch fetched above. This can be done by running

```
git checkout -b lab[n] origin/mlp2017-8/lab[n]
```

where again `lab[n]` corresponds to the relevant lab number fetched above e.g. `lab2`. This command creates a new local branch named `lab[n]` from the fetched branch on the remote repository `origin/mlp2017-8/lab[n]`.

Inside the `notebooks` directory there should new be a new notebook for today's lab. The notebook for the previous lab will now also have proposed solutions filled in.

To get started with the new notebook from the `~/mlpractical` directory start up a Jupyter notebook server

```
jupyter notebook
```

then open the new notebook from the dashboard.
