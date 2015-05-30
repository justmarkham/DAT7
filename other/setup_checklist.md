## Setup checklist

This is a checklist to confirm that your laptop is set up properly for DAT7. If at any point you get an error message, please note the error message and we will help you to fix it! If you don't get any error messages, you are properly set up.

Please post a message in the "setupchecklist" channel in Slack once you have walked through the entire checklist. That way, we will know whether or not we need to follow up with you.

### GitHub
* Log into your GitHub account, and "star" the DAT7 repository (the one you are looking at right now) by clicking the Star button in the upper right corner of the screen.

### Git
* Open a command line application (Git Bash on Windows, or Terminal on Mac).
* Type `git config --global user.name "YourFirstName YourLastName"` (including the quotes)
* Type `git config --global user.email "youremail@domain.com"` (use the email address associated with your GitHub account)
* Type `git clone https://github.com/justmarkham/DAT7.git`
    * This will create a folder called "DAT7" in the working directory of your local computer, and will copy the repository into that folder. (If you don't know what the working directory is, type `pwd`.)
    * You are welcome to move the DAT7 folder to another location on your computer, or delete it entirely. We will explain the purpose of cloning this repository during the course.

### Python
* While still at the command line:
    * Type `conda list` (if you choose not to use Anaconda, this will generate an error)
    * Type `pip install textblob`
    * Type `python` to open the Python interpreter
* While in the Python interpreter:
    * Look at the Python version number. It should start with 2.7. If your version number starts with 3, that's fine as long as you are aware of the differences between Python 2 and 3.
    * Type `import textblob`
    * Type `import pandas`
    * Type `exit()` to exit the interpreter. You can now close the command line application.
* Open Spyder (if you can't find Spyder, look for the Anaconda Launcher application)
    * In the console (probably on the right side of the screen), type `import textblob`
    * Type `import pandas`
