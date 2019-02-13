# Exceeded quota problems on DICE

Apologies to those who may have issues with having insufficient quota space on DICE in the labs on Monday (25th September).

This was caused by the [dynamic AFS quota system](http://computing.help.inf.ed.ac.uk/dynamic-afs-quotas) which only initially allocates users a subset of their maximum quota and then checks hourly to increase this quota as needed. Unfortunately the amount of disk space needed to store the temporary files used in installing the course dependencies exceeded the current dynamic quota for some people. This meant when running the `conda install ...` command it exited with a quota exceeded error.

Those who experienced that issue should now have sufficient quota space available. From any DICE computer, If you run in a terminal

```
source activate mlp
conda remove -y numpy scipy matplotlib jupyter
conda install -y numpy scipy matplotlib jupyter
conda clean -t -y
```

this should clean out the old partially installed packages and reinstall them from scratch which should now run to completion without a quota exceeded error.

Your homespace can be accessed from any Informatics computer running DICE (e.g. any of the computers in the [Forrest Hill labs](http://web.inf.ed.ac.uk/infweb/student-services/ito/students/year2/student-support/facilities/computer-labs) which are open-access outside of booked lab sessions or for those who know how to use SSH you can [log in remotely](http://computing.help.inf.ed.ac.uk/external-login)). You can therefore finish your environment set up prior to the next lab if you want though it is also fine to wait till the beginning of the next lab (it will take around 5 minutes to complete the installation).

At this point assuming you ran through the rest of the instructions to clone the Git repository to your homespace and install the `mlp` package (i.e. the instructions from [here](https://github.com/CSTR-Edinburgh/mlpractical/blob/mlp2016-7/lab1/environment-set-up.md#getting-the-course-code-and-a-short-introduction-to-git) on-wards), you should have a fully working environment.

Once your environment is set up in all future labs you will only need to activate it to get started. So at the beginning of each subsequent lab we will ask you to do something like the following

```
source activate mlp  # Activate the mlp environment
cd ~/mlpractical  # Change the current directory to mlpractical repository
git checkout mlp2017-8/lab[...]  # Checkout the branch for this week's lab
jupyter notebook  # Launch the notebook server
```
