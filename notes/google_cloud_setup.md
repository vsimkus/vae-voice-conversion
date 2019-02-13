# Google Cloud Usage Tutorial
This document has been created to help you setup a google cloud instance to be used for the MLP course using the student credit the course has acquired.
This document is non-exhaustive and many more useful information is available on the [google cloud documentation page](https://cloud.google.com/docs/).
For any question you might have, that is not covered here, a quick google search should get you what you need. Anything in the official google cloud docs should be very helpful.


### To create your account and start a project funded by the student credit

1. Get your coupon by following the instructions in the [coupon retrieval link](http://google.force.com/GCPEDU?cid=gUsa1mwo1yeUg4uWn5dJ7nsQXBQhsopl2fYBzL1ETQUEe7BiV6%2BQCOTy2nFd50d0/).
2. Once you receive your coupon, follow the email instructions to add your coupon to your account.
3. Once you have added your coupon, join the [MLPractical2018-9 Google Group](https://groups.google.com/d/forum/mlpractical_2018-9) using the same Google account you used to redeem your coupon. This ensures access to the shared disk images.
4. Once logged in, click on Projects (on the left hand side of the search bar on top of the page)
Name your project sxxxxxxx-MLPractical - replacing the sxxxxxxx with your student number.
5. Make sure that the financial source for your project is the MLPractical credit by clicking the 3 lines icon at the top left corner and then clicking billing -> go to linked billing account.
6. If it's not set to the MLPractical credits then set it by going to billing -> manage billing accounts -> My projects. Click the 3 dots under the Actions column for the relevant project and click change billing account. Select the MLPractical credit from your coupon.
6. Start the project

### To create an instance
1. Click the button with the three lines at the top left corner.
2. Click ```Compute Engine```
3. On the left hand side, click ```Images```
4. Click the ```CREATE IMAGE``` button at the top of the window
5. Name your image `mlpractical-2018-9-v14`
6. Set ```Source``` of image to be ```Image```
7. Set the ```Source project for images``` to be ```MLPractical``` with ID ```mlpractical-221202```. If you can't find this exact project ID then please post on Piazza to let us know and pause at this step until we resolve this for you.
8. Choose ```Source image``` to be ```mlpractical-2018-9-v14```.
9. Click ```Create``` and wait until it is done (this may take a few minutes).
10. Go back to the ```Images``` menu (this may happen automatically)
11. Tick in the box next to the ```mlpractical-2018-9-v14``` image that you have just created.
12. Click  ```CREATE INSTANCE```  at the top of the window
13. Name the instance ```mlpractical-1```
14. Select region to be ```us-west1``` and zone to be ```us-west-1b``` (there are other suitable regions however this one has K80s available right now so we went with this one, feel free to find something else if for some reason you need to, but it is recommended ro run on K80 GPUs.)
15. Select  ```2 vCPUs``` with ```7.5Gb memory```.
16. Click ```Customize```. Under ```GPUs``` select `1` of type ```NVidia Tesla K80```.
17. Disk size should be at least 30GB (feel free to add more, but it'll cost more). 
And the chosen image should be the `mlpractical-2018-9-v14`
18. Click ```Create```. 
Your instance should be ready in a minute or two.




#### Note
Be careful to select 1 x K80 GPU (P100s and P4s are 5x more expensive)

You only have $50 dollars worth of credit, which should be about 125 hours of GPU usage on a K80.


### To login into your instance via terminal:
1. In a DICE terminal window ```conda activate mlp```
2. Install the `gcloud` toolkit using ```conda install -c conda-forge google-cloud-sdk```
2. To authorize current machine to access your nodes run
```gcloud auth login```. This will authenticate your google account login.
3. Follow the prompts to get a token for your current machine.
4. Run ```gcloud config set project PROJECT_ID``` where you replace `PROJECT-ID` with your project ID, you can find that in the projects drop down menu on the top of the Google Compute Engine window; this sets the current project as the active one
5. In your compute engine window, in the line for the instance  that you have started (`mlpractical-1`), click on the downward arrow next to ```ssh```. Choose ```View gcloud command```. Copy the command to your terminal and press enter.
6. Add a password for your ssh-key (and remember it!). 
7. Re-enter password (which will unlock your ssh-key) when prompted.
8. Well done, you are now in your instance! When you login you may see an error of the form `Unable to set persistence mode for GPU 00000000:00:04.0: Insufficient Permissions` - you should be able to ignore this.  The instance on the first startup should check for the gpu cuda drivers and since they are not there, it will install them.  This will only happen once on your first login. Once the installation is finished you are ready to use the instance for your coursework.
9. Working in your Google Compute Instance that you are now logged into (via ssh), use ```conda activate mlp``` to activate your environment.
10. Run ```nvidia-smi``` to confirm that the GPU can be found.  This should report 1 Tesla K80 GPU.
11. Then clone a fresh mlpractical repository, and checkout branch `coursework_2`: 

```
git clone https://github.com/CSTR-Edinburgh/mlpractical.git ~/mlpractical
cd ~/mlpractical
git checkout -b coursework_2 origin/mlp2018-9/coursework_2
python setup.py develop
```

Then, to test PyTorch running on the GPU, run this script that trains a small convolutional network (4 conv layers, 64 filters) on EMNIST:
```
python mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py --dim_reduction_type max_pooling --use_gpu True
```
You should be able to see an experiment running, using the GPU. It should be doing about 60 it/s (iterations per second).  You can stop it when ever you like using `ctrl-c`.  

If all the above matches what’s stated then you should be ready to run your coursework jobs.

### Remember to ```stop``` your instance when not using it. You pay for the time you use the machine, not for the computational cycles used.
To stop the instance go to `Compute Engine -> VM instances` on the Google Cloud Platform, slect the instance and click ```Stop```.

#### Future ssh access:
To access the instance in the future simply run the `gcloud` command you copied from the google compute engine instance page.


## Copying data to and from an instance

Please look at the [google docs page on copying data](https://cloud.google.com/filestore/docs/copying-data).

To copy from local machine to a google instance, have a look at this [stackoverflow post](https://stackoverflow.com/questions/27857532/rsync-to-google-compute-engine-instance-from-jenkins).

## Running experiments over ssh:

If ssh fails while running an experiment, then the experiment is normally killed.
To avoid this use the command ```screen```. It creates a process of the current session that keeps running whether
 a user is signed in or not.
 
The basics of using screen is to use ```screen``` to create a new session, then to enter an existing session you use:
```screen -ls```
To get a list of all available sessions. Then once you find the one you want use:
```screen -d -r screen_id``` 
Replacing screen_id with the id of the session you want to enter.

While in a session, you can use 
- ```ctrl+a+esc``` To pause process and be able to scroll
- ```ctrl+a+d``` to detach from session while leaving it running (once you detach you can reattach using ```screen -r```)
- ```ctrl+a+n``` to see the next session.
- ```ctrl+a+c``` to create a new session
 
